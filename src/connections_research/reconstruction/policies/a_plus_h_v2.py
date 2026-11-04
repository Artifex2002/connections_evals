#!/usr/bin/env python3
"""
tier3_metrics_multiturn_a_plus_h.py
===================================

Deterministic Tier-3 metrics for multiturn NYT Connections.

Key semantics
-------------
1. Every Tier-1/Tier-2 candidate is retained as LOCAL hypothesis evidence.
2. A GLOBAL belief/partition state is reconstructed within each turn.
3. Hidden speculative belief resets at every turn boundary.
4. Officially solved groups are restored at turn start and are immutable.
5. A+H v2 handling for reasoning candidates:

   a) Reconstruct the candidate exactly as before (including refine_add/pivot).
   b) BEFORE stripping solved words, check whether the reconstructed unordered
      set exactly equals ANY previous submitted four-word guess in this run.
      Historical equality is METADATA, not an automatic blocker.
      If it is historical:
          - retain the full grouping as the LOCAL hypothesis and score its F1;
          - if it overlaps any officially solved word, do NOT mutate belief;
          - otherwise let CURRENT Tier-2 stance decide whether it is merely
            mentioned/rejected/enumerated or actively reconsidered. Held stances
            {propose, refine_add, refine_pivot, commit} may mutate belief.
   c) If it is NOT an exact historical guess, remove already-solved words.
   d) If fewer than 2 words remain, retain it as evidence but do not mutate
      belief.
   e) If 2-4 words remain and the Tier-2 stance is held, apply the effective
      hypothesis to belief.
   f) reject/mention/enumerate never mutate belief.
   g) Groups larger than 4 are retained/scored locally but do not mutate belief.

The solved-word lock in BeliefState is a hard invariant/defense-in-depth even
though A+H normally strips solved words before state mutation.

Local vs global scores
----------------------
- hypothesis_pair_f1: local score for the current EFFECTIVE hypothesis against
  its best matching gold group.
- pair_f1: global q_t score for the whole reconstructed active belief partition
  against the complete gold partition.

Resume behavior
---------------
The output stores a policy_version in processed_runs. A run is skipped only if:
    raw_turn_count, tier1_candidate_count, tier2_label_count, policy_version
all match. Thus a future Tier-3 policy change can force safe recomputation even
when Tier-1/Tier-2 row counts are unchanged.
"""

import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from itertools import combinations
from math import comb


HELD = {"propose", "refine_add", "refine_pivot", "commit"}
TURN_START_LABEL = "turn_start"
SUBMIT_LABEL = "submit"
POLICY_VERSION = "a_plus_h_v2_history_stance"


# =============================================================================
# BASIC PARSING
# =============================================================================

def load_json(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def parse_gold_groups(value):
    obj = load_json(value)
    if obj is None:
        return None

    if isinstance(obj, dict):
        iterable = list(obj.values())
    elif isinstance(obj, list):
        iterable = obj
    else:
        return None

    groups = []
    for item in iterable:
        if isinstance(item, dict):
            words = item.get("words")
            if not isinstance(words, list):
                continue
        elif isinstance(item, (list, tuple)):
            words = item
        else:
            continue

        group = {str(w).strip().upper() for w in words if str(w).strip()}
        groups.append(group)

    if len(groups) != 4 or any(len(g) != 4 for g in groups):
        return None
    if len(set().union(*groups)) != 16:
        return None
    return groups


def parse_guess(value):
    obj = load_json(value)
    if not isinstance(obj, list):
        return None
    words = [str(w).strip().upper() for w in obj if str(w).strip()]
    if len(words) != 4:
        return None
    group = set(words)
    return group if len(group) == 4 else None


def parse_solved_indices(value):
    obj = load_json(value)
    if not isinstance(obj, list):
        return []
    out = []
    for x in obj:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < 4:
            out.append(i)
    return sorted(set(out))


# =============================================================================
# HYPOTHESIS RECONSTRUCTION
# =============================================================================

def reconstruct_turn(candidates):
    """
    Original within-turn reconstruction semantics.

    IMPORTANT: previous_candidate is the immediately preceding reconstructed
    Tier-1 candidate, NOT the previous belief-updating candidate. This remains
    aligned with the Tier-2 prompt/classifier design.
    """
    previous_candidate = set()
    output = []

    for candidate in candidates:
        raw = set(candidate["raw"])
        label = candidate["label"]

        if label == "refine_pivot":
            resolved = raw
        elif label == "refine_add":
            if previous_candidate and not (raw & previous_candidate):
                resolved = raw
            elif previous_candidate and raw < previous_candidate:
                resolved = raw
            else:
                resolved = previous_candidate | raw
        else:
            resolved = raw

        output.append({
            **candidate,
            "prestrip_resolved": set(resolved),
            "kept": label in HELD,
        })
        previous_candidate = set(resolved)

    return output


def apply_a_plus_h(candidate, solved_words, previous_guess_sets):
    """
    Apply A+H v2 to ONE reconstructed reasoning candidate.

    Every candidate remains LOCAL hypothesis evidence. Historical equality with
    a previous official guess is metadata, not an automatic blocker.

    Historical exact match:
      * score the FULL historical grouping locally;
      * if it overlaps an officially solved word, never mutate belief;
      * otherwise use CURRENT stance: held stances may actively reconsider and
        mutate belief, non-held stances remain review/rejection evidence only.

    Non-historical candidate:
      * strip officially solved words;
      * require an effective core of 2-4 words;
      * only held stances may mutate belief.

    The returned `resolved` set is what receives the LOCAL hypothesis score.
    `belief_update_allowed` controls the GLOBAL belief transition.
    """
    prestrip = set(candidate["prestrip_resolved"])
    solved_overlap = prestrip & solved_words
    historical_match = frozenset(prestrip) in previous_guess_sets

    if historical_match:
        # Preserve the entire old official guess as the local hypothesis.
        effective = set(prestrip)
        stripped = False

        # Once any word belongs to an authoritative solved group, the old
        # four-word set cannot be reinstated as an active game-state grouping.
        if solved_overlap:
            blocked_reason = "historical_guess_solved_overlap"
            belief_update_allowed = False
        elif len(effective) < 2:
            blocked_reason = "too_small"
            belief_update_allowed = False
        elif len(effective) > 4:
            blocked_reason = "too_large"
            belief_update_allowed = False
        elif candidate["label"] not in HELD:
            blocked_reason = "nonheld_stance"
            belief_update_allowed = False
        else:
            # Genuine CURRENT reconsideration of an earlier official attempt.
            blocked_reason = None
            belief_update_allowed = True

    else:
        effective = set(prestrip) - solved_words
        stripped = bool(solved_overlap)

        if len(effective) < 2:
            blocked_reason = "too_small_after_solved_strip" if stripped else "too_small"
            belief_update_allowed = False
        elif len(effective) > 4:
            blocked_reason = "too_large"
            belief_update_allowed = False
        elif candidate["label"] not in HELD:
            blocked_reason = "nonheld_stance"
            belief_update_allowed = False
        else:
            blocked_reason = None
            belief_update_allowed = True

    well_formed = 2 <= len(effective) <= 4

    return {
        **candidate,
        "resolved": set(effective),
        "solved_overlap": set(solved_overlap),
        "historical_guess_match": historical_match,
        "stripped_solved_words": stripped,
        "well_formed": well_formed,
        "belief_update_allowed": belief_update_allowed,
        "blocked_reason": blocked_reason,
    }


# =============================================================================
# PAIRWISE QUALITY
# =============================================================================

def pairs_for_group(group):
    if len(group) < 2:
        return set()
    return {tuple(sorted(p)) for p in combinations(sorted(group), 2)}


def gold_pairs_for_partition(gold):
    pairs = set()
    for g in gold:
        pairs |= pairs_for_group(g)
    return pairs


def pairwise_prf(partition, gold_pairs):
    predicted_pairs = set()
    for group in partition:
        predicted_pairs |= pairs_for_group(group)

    if not predicted_pairs:
        return 0.0, 0.0, 0.0

    inter = len(predicted_pairs & gold_pairs)
    precision = inter / len(predicted_pairs)
    recall = inter / len(gold_pairs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def hypothesis_best_gold_score(hypothesis, gold):
    """Local pairwise F1 of one hypothesis vs its best matching gold group."""
    predicted_pairs = pairs_for_group(hypothesis)
    if not predicted_pairs:
        return 0.0, None

    best_f1, best_index = 0.0, None
    for gi, gold_group in enumerate(gold):
        target_pairs = pairs_for_group(gold_group)
        inter = len(predicted_pairs & target_pairs)
        precision = inter / len(predicted_pairs)
        recall = inter / len(target_pairs)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1, best_index = f1, gi

    return round(best_f1, 4), best_index


# =============================================================================
# PARTITION QUALITY
# =============================================================================

def exact_hits(partition, gold):
    gold_sets = {frozenset(g) for g in gold}
    return sum(1 for g in partition if frozenset(g) in gold_sets)


def adjusted_rand(partition, gold):
    if len(partition) != 4 or any(len(g) != 4 for g in partition):
        return None

    gold_words = set().union(*gold)
    predicted_words = set().union(*partition)
    if predicted_words != gold_words or sum(len(g) for g in partition) != 16:
        return None

    gold_id = {w: gi for gi, g in enumerate(gold) for w in g}
    pred_id = {w: gi for gi, g in enumerate(partition) for w in g}
    words = sorted(gold_words)
    gold_labels = [gold_id[w] for w in words]
    pred_labels = [pred_id[w] for w in words]
    n = len(words)

    contingency_index = sum(comb(c, 2) for c in Counter(zip(gold_labels, pred_labels)).values())
    gold_sum = sum(comb(c, 2) for c in Counter(gold_labels).values())
    pred_sum = sum(comb(c, 2) for c in Counter(pred_labels).values())
    expected = gold_sum * pred_sum / comb(n, 2)
    maximum = (gold_sum + pred_sum) / 2
    return 1.0 if maximum == expected else (contingency_index - expected) / (maximum - expected)


# =============================================================================
# BELIEF STATE
# =============================================================================

class BeliefState:
    """
    Last-writer-wins mutable belief state WITHIN a turn.

    Solved groups are inserted with lock=True and become immutable. This is a
    hard game-state invariant and defense-in-depth for A+H.
    """
    def __init__(self):
        self.word_to_slot = {}
        self.slots = []
        self.locked_words = set()

    def apply_group(self, group, *, lock=False):
        group = set(group)
        detached = []

        if not lock:
            group = {w for w in group if w not in self.locked_words}

        if not group:
            return detached

        for word in group:
            old_slot = self.word_to_slot.get(word)
            if old_slot is not None and word in old_slot and word not in self.locked_words:
                old_slot.discard(word)
                detached.append(word)

        new_slot = set(group)
        self.slots.append(new_slot)
        for word in group:
            self.word_to_slot[word] = new_slot

        self.slots = [slot for slot in self.slots if slot]

        if lock:
            self.locked_words.update(group)

        return detached

    def partition(self):
        return [set(slot) for slot in self.slots if len(slot) >= 2]

    def n_assigned(self):
        return sum(len(slot) for slot in self.slots)

    def max_group_size(self):
        return max((len(slot) for slot in self.slots), default=0)


# =============================================================================
# STEP SCORING
# =============================================================================

def score_state(*, state, gold, gold_pairs, hypothesis):
    partition = state.partition()
    precision, recall, f1 = pairwise_prf(partition, gold_pairs)
    hypothesis_f1, best_gold_index = hypothesis_best_gold_score(hypothesis, gold)

    return {
        "partition": partition,
        "pair_p": round(precision, 4),
        "pair_r": round(recall, 4),
        "pair_f1": round(f1, 4),
        "hypothesis_pair_f1": hypothesis_f1,
        "hypothesis_best_gold_index": best_gold_index,
        "exact_hits": exact_hits(partition, gold),
        "ari": adjusted_rand(partition, gold),
        "n_assigned": state.n_assigned(),
        "n_groups": len(partition),
        "max_group_size": state.max_group_size(),
    }


# =============================================================================
# BUILD MULTITURN TRAJECTORY
# =============================================================================

def build_run_series(*, raw_turns, candidates_by_turn, labels_by_turn, gold):
    gold_pairs = gold_pairs_for_partition(gold)
    vocabulary = set().union(*gold)
    series = []
    global_step = 0

    # Exact unordered official guesses from STRICTLY EARLIER turns.
    previous_guess_sets = set()

    for turn in raw_turns:
        turn_id = turn["turn_id"]
        turn_index = turn["turn_index"]

        state = BeliefState()
        solved_before = parse_solved_indices(turn["solved_group_indices_before_json"])
        solved_words = set()

        for gi in solved_before:
            solved_words |= set(gold[gi])
            state.apply_group(gold[gi], lock=True)

        # Explicit turn boundary / authoritative game state.
        scored = score_state(state=state, gold=gold, gold_pairs=gold_pairs, hypothesis=set())
        series.append({
            "global_step": global_step,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "candidate_index": None,
            "step_type": "turn_start",
            "text_source": "game_state",
            "label": TURN_START_LABEL,
            "guess_status": None,
            "kept": 0,
            "well_formed": 1,
            "belief_update_applied": 0,
            "historical_guess_match": 0,
            "stripped_solved_words": 0,
            "blocked_reason": None,
            "raw": [],
            "prestrip_resolved": [],
            "resolved": [],
            "solved_overlap": [],
            "detached": [],
            "solved_before": solved_before,
            **scored,
        })
        global_step += 1

        tier1_candidates = candidates_by_turn.get(turn_id, [])
        tier2_labels = labels_by_turn.get(turn_id, {})

        if tier1_candidates:
            candidate_indexes = {c["candidate_index"] for c in tier1_candidates}
            label_indexes = set(tier2_labels)
            if candidate_indexes != label_indexes:
                return None, (
                    f"Tier-2 coverage mismatch for turn {turn_id}: "
                    f"{len(candidate_indexes)} candidates vs {len(label_indexes)} labels"
                )

        reconstruction_input = []
        for c in tier1_candidates:
            ci = c["candidate_index"]
            reconstruction_input.append({
                "candidate_index": ci,
                "raw": set(c["raw"]),
                "label": tier2_labels[ci],
                "text_source": c["text_source"],
            })

        reconstructed = reconstruct_turn(reconstruction_input)

        for candidate in reconstructed:
            candidate = apply_a_plus_h(candidate, solved_words, previous_guess_sets)
            detached = []
            applied = False

            if candidate["belief_update_allowed"]:
                detached = state.apply_group(candidate["resolved"])
                applied = True

            # LOCAL hypothesis F1 uses A+H's effective hypothesis:
            # - full prior guess if historical exact-match;
            # - otherwise solved words stripped from the reconstructed candidate.
            scored = score_state(
                state=state,
                gold=gold,
                gold_pairs=gold_pairs,
                hypothesis=candidate["resolved"],
            )

            series.append({
                "global_step": global_step,
                "turn_id": turn_id,
                "turn_index": turn_index,
                "candidate_index": candidate["candidate_index"],
                "step_type": "reasoning",
                "text_source": candidate["text_source"],
                "label": candidate["label"],
                "guess_status": None,
                "kept": int(candidate["kept"]),
                "well_formed": int(candidate["well_formed"]),
                "belief_update_applied": int(applied),
                "historical_guess_match": int(candidate["historical_guess_match"]),
                "stripped_solved_words": int(candidate["stripped_solved_words"]),
                "blocked_reason": candidate["blocked_reason"],
                "raw": sorted(candidate["raw"]),
                "prestrip_resolved": sorted(candidate["prestrip_resolved"]),
                "resolved": sorted(candidate["resolved"]),
                "solved_overlap": sorted(candidate["solved_overlap"]),
                "detached": sorted(detached),
                "solved_before": solved_before,
                **scored,
            })
            global_step += 1

        # Actual submitted action is an action anchor, not a reasoning candidate.
        submitted_group = parse_guess(turn["parsed_guess_json"])
        if turn["parse_success"] and submitted_group and submitted_group <= vocabulary:
            detached = state.apply_group(submitted_group)
            scored = score_state(
                state=state,
                gold=gold,
                gold_pairs=gold_pairs,
                hypothesis=submitted_group,
            )
            series.append({
                "global_step": global_step,
                "turn_id": turn_id,
                "turn_index": turn_index,
                "candidate_index": None,
                "step_type": "submit",
                "text_source": "action",
                "label": SUBMIT_LABEL,
                "guess_status": turn["guess_status"],
                "kept": 1,
                "well_formed": 1,
                "belief_update_applied": 1,
                "historical_guess_match": 0,
                "stripped_solved_words": 0,
                "blocked_reason": None,
                "raw": sorted(submitted_group),
                "prestrip_resolved": sorted(submitted_group),
                "resolved": sorted(submitted_group),
                "solved_overlap": sorted(submitted_group & solved_words),
                "detached": sorted(detached),
                "solved_before": solved_before,
                **scored,
            })
            global_step += 1

            # Becomes historical only AFTER this turn's reasoning/action.
            previous_guess_sets.add(frozenset(submitted_group))

    return series, None


# =============================================================================
# STATE HELPERS + M1
# =============================================================================

def word_to_group(partition):
    mapping = {}
    for group in partition:
        frozen = frozenset(group)
        if len(frozen) >= 2:
            for word in frozen:
                mapping[word] = frozen
    return mapping


def final_partition(series):
    if not series:
        return []
    return [frozenset(g) for g in series[-1]["partition"]]


def churn_for_series(series):
    signature_maps = [word_to_group(step["partition"]) for step in series]
    if not signature_maps:
        return 0, 0

    words = set().union(*[set(m) for m in signature_maps])
    total = 0
    for word in words:
        previous = None
        for mapping in signature_maps:
            current = mapping.get(word)
            if current != previous:
                if previous is not None and (
                    current is None or not (current >= previous or current <= previous)
                ):
                    total += 1
                previous = current

    return total, max(len(series) - 1, 0)


def m1_run_churn_components(series):
    by_turn = defaultdict(list)
    for step in series:
        by_turn[step["turn_index"]].append(step)

    total = transitions = 0
    for turn_index in sorted(by_turn):
        t, tr = churn_for_series(by_turn[turn_index])
        total += t
        transitions += tr

    return {
        "total": total,
        "per_word": round(total / 16.0, 4),
        "per_word_step": round(total / (16.0 * transitions), 6) if transitions else 0.0,
        "transitions": transitions,
    }


def m1_correct_abandonment(series, gold):
    if not series:
        return 0.0
    final = {frozenset(g) for g in series[-1]["partition"]}
    appeared_before = {
        frozenset(g)
        for step in series[:-1]
        for g in step["partition"]
    }
    abandoned = sum(
        1 for gold_group in gold
        if frozenset(gold_group) in appeared_before and frozenset(gold_group) not in final
    )
    return round(abandoned / len(gold), 4)


def m1_lockin_latencies(series):
    if not series:
        return []
    final_groups = final_partition(series)
    if not final_groups:
        return []
    if len(series) <= 1:
        return [0.0 for _ in final_groups]

    per_step = [{frozenset(g) for g in step["partition"]} for step in series]
    latencies = []
    for final_group in final_groups:
        index = len(series) - 1
        lock_index = index
        while index >= 0 and final_group in per_step[index]:
            lock_index = index
            index -= 1
        latencies.append(round(lock_index / (len(series) - 1), 4))
    return latencies


# =============================================================================
# M2
# =============================================================================

def m2_components(series, gold):
    gold_index = {w: gi for gi, g in enumerate(gold) for w in g}
    blocked_blast_sum = wrong_word_steps = blocked_word_steps = 0
    total_blast_on_wrong = max_blast = 0

    for step in series:
        mapping = word_to_group(step["partition"])
        for word, slot in mapping.items():
            gi = gold_index.get(word)
            if gi is None:
                continue
            blast = sum(1 for other in slot if gold_index.get(other) != gi)
            if blast == 0:
                continue

            wrong_word_steps += 1
            total_blast_on_wrong += blast
            max_blast = max(max_blast, blast)
            gold_mates = [mate for mate in gold[gi] if mate != word]
            blocked = any(mate in mapping and mate not in slot for mate in gold_mates)
            if blocked:
                blocked_word_steps += 1
                blocked_blast_sum += blast

    n_steps = len(series)
    blast_duration = blocked_blast_sum / (16 * n_steps) if n_steps else 0.0
    mean_blast = total_blast_on_wrong / wrong_word_steps if wrong_word_steps else 0.0

    return {
        "m2_blast_duration": round(blast_duration, 6),
        "m2_wrong_word_steps": wrong_word_steps,
        "m2_blocked_word_steps": blocked_word_steps,
        "m2_blocked_blast_sum": blocked_blast_sum,
        "m2_mean_blast_radius": round(mean_blast, 4),
        "m2_max_blast_radius": max_blast,
    }


# =============================================================================
# M3 STRUCTURAL DIAGNOSTIC
# =============================================================================

def m3_dstruct(series, gold):
    if not series:
        return 16, 16, 4, 16

    final = [set(g) for g in series[-1]["partition"]]
    assigned = set().union(*final) if final else set()
    all_words = set().union(*gold)
    word_violations = len(all_words - assigned)
    group_violations = abs(len(final) - 4)
    size_violations = sum(abs(len(g) - 4) for g in final)
    total = word_violations + group_violations + size_violations
    return total, word_violations, group_violations, size_violations


# =============================================================================
# RUN-LEVEL METRICS
# =============================================================================

def compute_run_metrics(*, series, gold, source_run):
    if not series:
        return None

    qualities = [step["pair_f1"] for step in series]
    peak = max(qualities)
    final = qualities[-1]
    peak_index = qualities.index(peak)
    peak_fraction = peak_index / (len(qualities) - 1) if len(qualities) > 1 else 0.0

    churn = m1_run_churn_components(series)
    lock_latencies = m1_lockin_latencies(series)
    m2 = m2_components(series, gold)
    m3_total, m3_word, m3_group, m3_size = m3_dstruct(series, gold)

    reasoning = [s for s in series if s["step_type"] == "reasoning"]

    return {
        "n_steps": len(series),
        "n_turn_starts": sum(s["step_type"] == "turn_start" for s in series),
        "n_candidates": len(reasoning),
        "n_kept": sum(s["kept"] and s["well_formed"] for s in reasoning),
        "n_action_anchors": sum(s["step_type"] == "submit" for s in series),
        "n_belief_updates": sum(s["belief_update_applied"] for s in reasoning),
        "n_historical_guess_matches": sum(s["historical_guess_match"] for s in reasoning),
        "n_solved_overlap_candidates": sum(bool(s["solved_overlap"]) for s in reasoning),
        "n_stripped_candidates": sum(s["stripped_solved_words"] for s in reasoning),
        "n_blocked_small_core": sum(
            s["blocked_reason"] in {"too_small_after_solved_strip", "too_small"}
            for s in reasoning
        ),

        "m4_regret": round(peak - final, 4),
        "m4_peak": round(peak, 4),
        "m4_final": round(final, 4),
        "m4_peak_frac": round(peak_fraction, 4),
        "m4_peak_global_step": peak_index,
        "m4_peak_turn_index": series[peak_index]["turn_index"],

        "m1_churn_total": churn["total"],
        "m1_churn_per_word": churn["per_word"],
        "m1_churn_per_word_step": churn["per_word_step"],
        "m1_churn_transitions": churn["transitions"],
        "m1_correct_abandonment": m1_correct_abandonment(series, gold),
        "m1_lockin_json": json.dumps(lock_latencies),
        "m1_lockin_mean": round(sum(lock_latencies) / len(lock_latencies), 4) if lock_latencies else None,

        **m2,

        "m3_dstruct": m3_total,
        "m3_word_violations": m3_word,
        "m3_group_violations": m3_group,
        "m3_size_violations": m3_size,

        "final_exact_groups": series[-1]["exact_hits"],
        "final_ari": series[-1]["ari"],
        "trajectory_solved": int(series[-1]["exact_hits"] == 4),

        "db_solved": source_run["solved"],
        "db_binary_score": source_run["binary_score"],
        "guesses_used": source_run["guesses_used"],
        "mistakes_used": source_run["mistakes_used"],
    }


# =============================================================================
# TURN-LEVEL METRICS
# =============================================================================

def compute_turn_metrics(*, series, raw_turns):
    steps_by_turn = defaultdict(list)
    for step in series:
        steps_by_turn[step["turn_index"]].append(step)

    raw_by_turn = {t["turn_index"]: t for t in raw_turns}
    summaries = []

    for turn_index in sorted(raw_by_turn):
        raw_turn = raw_by_turn[turn_index]
        local = steps_by_turn.get(turn_index, [])
        starts = [s for s in local if s["step_type"] == "turn_start"]
        reasoning = [s for s in local if s["step_type"] == "reasoning"]
        submits = [s for s in local if s["step_type"] == "submit"]

        q_start = starts[0]["pair_f1"] if starts else 0.0
        q_first = reasoning[0]["pair_f1"] if reasoning else (submits[0]["pair_f1"] if submits else q_start)
        q_pre_submit = reasoning[-1]["pair_f1"] if reasoning else q_start
        q_submit = submits[-1]["pair_f1"] if submits else None
        q_end = local[-1]["pair_f1"] if local else q_start
        q_peak = max([s["pair_f1"] for s in local] or [q_start])
        churn_total, churn_transitions = churn_for_series(local)

        summaries.append({
            "run_id": raw_turn["run_id"],
            "puzzle_id": raw_turn["puzzle_id"],
            "model": raw_turn["model"],
            "turn_id": raw_turn["turn_id"],
            "turn_index": turn_index,
            "guess_status": raw_turn["guess_status"],
            "feedback_returned": raw_turn["feedback_returned"],
            "parse_success": raw_turn["parse_success"],
            "solved_before_json": raw_turn["solved_group_indices_before_json"],
            "solved_after_json": raw_turn["solved_group_indices_after_json"],
            "n_candidates": len(reasoning),
            "n_kept": sum(s["kept"] and s["well_formed"] for s in reasoning),
            "n_belief_updates": sum(s["belief_update_applied"] for s in reasoning),
            "n_historical_guess_matches": sum(s["historical_guess_match"] for s in reasoning),
            "n_solved_overlap_candidates": sum(bool(s["solved_overlap"]) for s in reasoning),
            "n_stripped_candidates": sum(s["stripped_solved_words"] for s in reasoning),
            "n_blocked_small_core": sum(
                s["blocked_reason"] in {"too_small_after_solved_strip", "too_small"}
                for s in reasoning
            ),
            "has_submit_anchor": int(bool(submits)),
            "q_start": round(q_start, 4),
            "q_first": round(q_first, 4),
            "q_pre_submit": round(q_pre_submit, 4),
            "q_submit": round(q_submit, 4) if q_submit is not None else None,
            "q_end": round(q_end, 4),
            "q_peak": round(q_peak, 4),
            "q_regret": round(q_peak - q_end, 4),
            "within_turn_gain": round(q_pre_submit - q_start, 4),
            "action_gain": round(q_submit - q_pre_submit, 4) if q_submit is not None else None,
            "context_reset_delta": None,
            "next_turn_initial_gain": None,
            "response_delta": None,
            "m1_turn_churn_total": churn_total,
            "m1_turn_churn_per_word": round(churn_total / 16.0, 4),
            "m1_turn_churn_per_word_step": (
                round(churn_total / (16.0 * churn_transitions), 6) if churn_transitions else 0.0
            ),
        })

    for i in range(len(summaries) - 1):
        cur, nxt = summaries[i], summaries[i + 1]
        cur["context_reset_delta"] = round(nxt["q_start"] - cur["q_end"], 4)
        cur["next_turn_initial_gain"] = round(nxt["q_first"] - nxt["q_start"], 4)
        cur["response_delta"] = round(nxt["q_first"] - cur["q_end"], 4)

    return summaries


# =============================================================================
# OUTPUT DATABASE
# =============================================================================

def ensure_column(conn, table, column, definition):
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def ensure_out(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_metrics(
            run_id TEXT PRIMARY KEY,
            puzzle_id TEXT,
            model TEXT,
            n_turns INTEGER,
            n_steps INTEGER,
            n_turn_starts INTEGER,
            n_candidates INTEGER,
            n_kept INTEGER,
            n_action_anchors INTEGER,
            n_belief_updates INTEGER,
            n_historical_guess_matches INTEGER,
            n_solved_overlap_candidates INTEGER,
            n_stripped_candidates INTEGER,
            n_blocked_small_core INTEGER,
            m4_regret REAL,
            m4_peak REAL,
            m4_final REAL,
            m4_peak_frac REAL,
            m4_peak_global_step INTEGER,
            m4_peak_turn_index INTEGER,
            m1_churn_total INTEGER,
            m1_churn_per_word REAL,
            m1_churn_per_word_step REAL,
            m1_churn_transitions INTEGER,
            m1_correct_abandonment REAL,
            m1_lockin_mean REAL,
            m1_lockin_json TEXT,
            m2_blast_duration REAL,
            m2_wrong_word_steps INTEGER,
            m2_blocked_word_steps INTEGER,
            m2_blocked_blast_sum INTEGER,
            m2_mean_blast_radius REAL,
            m2_max_blast_radius INTEGER,
            m3_dstruct INTEGER,
            m3_word_violations INTEGER,
            m3_group_violations INTEGER,
            m3_size_violations INTEGER,
            final_exact_groups INTEGER,
            final_ari REAL,
            trajectory_solved INTEGER,
            db_solved INTEGER,
            db_binary_score INTEGER,
            guesses_used INTEGER,
            mistakes_used INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS turn_metrics(
            run_id TEXT,
            puzzle_id TEXT,
            model TEXT,
            turn_id TEXT,
            turn_index INTEGER,
            guess_status TEXT,
            feedback_returned TEXT,
            parse_success INTEGER,
            solved_before_json TEXT,
            solved_after_json TEXT,
            n_candidates INTEGER,
            n_kept INTEGER,
            n_belief_updates INTEGER,
            n_historical_guess_matches INTEGER,
            n_solved_overlap_candidates INTEGER,
            n_stripped_candidates INTEGER,
            n_blocked_small_core INTEGER,
            has_submit_anchor INTEGER,
            q_start REAL,
            q_first REAL,
            q_pre_submit REAL,
            q_submit REAL,
            q_end REAL,
            q_peak REAL,
            q_regret REAL,
            within_turn_gain REAL,
            action_gain REAL,
            context_reset_delta REAL,
            next_turn_initial_gain REAL,
            response_delta REAL,
            m1_turn_churn_total INTEGER,
            m1_turn_churn_per_word REAL,
            m1_turn_churn_per_word_step REAL,
            PRIMARY KEY(run_id, turn_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS step_cache(
            run_id TEXT,
            puzzle_id TEXT,
            model TEXT,
            turn_id TEXT,
            turn_index INTEGER,
            global_step INTEGER,
            candidate_index INTEGER,
            step_type TEXT,
            text_source TEXT,
            label TEXT,
            guess_status TEXT,
            kept INTEGER,
            well_formed INTEGER,
            belief_update_applied INTEGER,
            historical_guess_match INTEGER,
            stripped_solved_words INTEGER,
            blocked_reason TEXT,
            solved_before_json TEXT,
            raw_json TEXT,
            prestrip_resolved_json TEXT,
            resolved_json TEXT,
            solved_overlap_json TEXT,
            partition_json TEXT,
            detached_json TEXT,
            hypothesis_pair_f1 REAL,
            hypothesis_best_gold_index INTEGER,
            pair_p REAL,
            pair_r REAL,
            pair_f1 REAL,
            exact_hits INTEGER,
            ari REAL,
            n_assigned INTEGER,
            n_groups INTEGER,
            max_group_size INTEGER,
            PRIMARY KEY(run_id, global_step)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_runs(
            run_id TEXT PRIMARY KEY,
            source_db TEXT,
            raw_turn_count INTEGER,
            tier1_candidate_count INTEGER,
            tier2_label_count INTEGER,
            policy_version TEXT,
            processed_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS skipped_runs(
            run_id TEXT PRIMARY KEY,
            source_db TEXT,
            reason TEXT,
            last_attempted_at TEXT
        )
    """)

    # Backward-compatible schema upgrades if an older output DB is reused.
    for col, definition in [
        ("n_belief_updates", "INTEGER"),
        ("n_historical_guess_matches", "INTEGER"),
        ("n_solved_overlap_candidates", "INTEGER"),
        ("n_stripped_candidates", "INTEGER"),
        ("n_blocked_small_core", "INTEGER"),
    ]:
        ensure_column(conn, "run_metrics", col, definition)
        ensure_column(conn, "turn_metrics", col, definition)

    for col, definition in [
        ("belief_update_applied", "INTEGER"),
        ("historical_guess_match", "INTEGER"),
        ("stripped_solved_words", "INTEGER"),
        ("blocked_reason", "TEXT"),
        ("prestrip_resolved_json", "TEXT"),
        ("solved_overlap_json", "TEXT"),
    ]:
        ensure_column(conn, "step_cache", col, definition)

    ensure_column(conn, "processed_runs", "policy_version", "TEXT")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_step_run_turn ON step_cache(run_id, turn_index, global_step)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turn_model ON turn_metrics(model, turn_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_model ON run_metrics(model)")
    conn.commit()


# =============================================================================
# RESUME HELPERS
# =============================================================================

def get_processed_signature(out, run_id):
    row = out.execute("""
        SELECT raw_turn_count, tier1_candidate_count, tier2_label_count, policy_version
        FROM processed_runs WHERE run_id = ?
    """, (run_id,)).fetchone()
    return tuple(row) if row else None


def current_signature(*, source, tier1, tier2, run_id):
    raw_turn_count = source.execute("SELECT COUNT(*) FROM turns WHERE run_id = ?", (run_id,)).fetchone()[0]
    tier1_count = tier1.execute("SELECT COUNT(*) FROM tier1_candidates WHERE run_id = ?", (run_id,)).fetchone()[0]
    tier2_count = tier2.execute("SELECT COUNT(*) FROM tier2_labels WHERE run_id = ?", (run_id,)).fetchone()[0]
    return raw_turn_count, tier1_count, tier2_count, POLICY_VERSION


def clear_run_output(out, run_id):
    for table in ["step_cache", "turn_metrics", "run_metrics", "processed_runs", "skipped_runs"]:
        out.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
    out.commit()


def record_skip(out, *, run_id, source_db, reason):
    out.execute("INSERT OR REPLACE INTO skipped_runs VALUES (?, ?, ?, ?)", (
        run_id, source_db, reason, time.strftime("%Y-%m-%dT%H:%M:%S")
    ))
    out.commit()


# =============================================================================
# LOADERS
# =============================================================================

def load_tier1_candidates(tier1, run_id):
    rows = tier1.execute("""
        SELECT turn_id, candidate_index, wordset_json, text_source
        FROM tier1_candidates
        WHERE run_id = ?
        ORDER BY turn_index, candidate_index
    """, (run_id,)).fetchall()

    by_turn = defaultdict(list)
    for turn_id, ci, wordset_json, text_source in rows:
        raw = load_json(wordset_json)
        if not isinstance(raw, list):
            raw = []
        by_turn[turn_id].append({
            "candidate_index": ci,
            "raw": {str(w).strip().upper() for w in raw if str(w).strip()},
            "text_source": text_source,
        })
    return by_turn


def load_tier2_labels(tier2, run_id):
    rows = tier2.execute("""
        SELECT turn_id, candidate_index, label
        FROM tier2_labels
        WHERE run_id = ?
        ORDER BY turn_index, candidate_index
    """, (run_id,)).fetchall()

    by_turn = defaultdict(dict)
    for turn_id, ci, label in rows:
        by_turn[turn_id][ci] = label
    return by_turn


def source_runs(conn, model_filter=None):
    query = """
        SELECT
            r.run_id, r.puzzle_id, r.model_name,
            r.solved, r.binary_score, r.guesses_used, r.mistakes_used,
            p.ground_truth_json
        FROM runs r
        JOIN puzzles p ON p.puzzle_id = r.puzzle_id
        WHERE r.status = 'completed'
    """
    params = []
    if model_filter:
        query += " AND r.model_name = ?"
        params.append(model_filter)
    query += " ORDER BY r.run_id"
    return conn.execute(query, params)


def load_raw_turns(conn, *, run_id, puzzle_id, model):
    rows = conn.execute("""
        SELECT
            turn_id, turn_index,
            parsed_guess_json, guess_status, feedback_returned, parse_success,
            solved_group_indices_before_json, solved_group_indices_after_json
        FROM turns
        WHERE run_id = ?
        ORDER BY turn_index
    """, (run_id,)).fetchall()

    return [{
        "run_id": run_id,
        "puzzle_id": puzzle_id,
        "model": model,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "parsed_guess_json": parsed_guess_json,
        "guess_status": guess_status,
        "feedback_returned": feedback_returned,
        "parse_success": parse_success,
        "solved_group_indices_before_json": solved_before,
        "solved_group_indices_after_json": solved_after,
    } for (
        turn_id, turn_index, parsed_guess_json, guess_status, feedback_returned,
        parse_success, solved_before, solved_after
    ) in rows]


# =============================================================================
# WRITERS
# =============================================================================

def write_step_cache(out, *, run_id, puzzle_id, model, series):
    rows = []
    for s in series:
        rows.append((
            run_id, puzzle_id, model,
            s["turn_id"], s["turn_index"], s["global_step"], s["candidate_index"],
            s["step_type"], s["text_source"], s["label"], s["guess_status"],
            s["kept"], s["well_formed"], s["belief_update_applied"],
            s["historical_guess_match"], s["stripped_solved_words"], s["blocked_reason"],
            json.dumps(s["solved_before"]),
            json.dumps(s["raw"]),
            json.dumps(s["prestrip_resolved"]),
            json.dumps(s["resolved"]),
            json.dumps(s["solved_overlap"]),
            json.dumps([sorted(g) for g in s["partition"]]),
            json.dumps(s["detached"]),
            s["hypothesis_pair_f1"], s["hypothesis_best_gold_index"],
            s["pair_p"], s["pair_r"], s["pair_f1"],
            s["exact_hits"], s["ari"], s["n_assigned"], s["n_groups"], s["max_group_size"],
        ))

    out.executemany("""
        INSERT OR REPLACE INTO step_cache(
            run_id, puzzle_id, model,
            turn_id, turn_index, global_step, candidate_index,
            step_type, text_source, label, guess_status,
            kept, well_formed, belief_update_applied,
            historical_guess_match, stripped_solved_words, blocked_reason,
            solved_before_json,
            raw_json, prestrip_resolved_json, resolved_json, solved_overlap_json,
            partition_json, detached_json,
            hypothesis_pair_f1, hypothesis_best_gold_index,
            pair_p, pair_r, pair_f1,
            exact_hits, ari, n_assigned, n_groups, max_group_size
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, rows)


def write_turn_metrics(out, summaries):
    rows = []
    for s in summaries:
        rows.append((
            s["run_id"], s["puzzle_id"], s["model"], s["turn_id"], s["turn_index"],
            s["guess_status"], s["feedback_returned"], s["parse_success"],
            s["solved_before_json"], s["solved_after_json"],
            s["n_candidates"], s["n_kept"], s["n_belief_updates"],
            s["n_historical_guess_matches"], s["n_solved_overlap_candidates"],
            s["n_stripped_candidates"], s["n_blocked_small_core"], s["has_submit_anchor"],
            s["q_start"], s["q_first"], s["q_pre_submit"], s["q_submit"], s["q_end"],
            s["q_peak"], s["q_regret"], s["within_turn_gain"], s["action_gain"],
            s["context_reset_delta"], s["next_turn_initial_gain"], s["response_delta"],
            s["m1_turn_churn_total"], s["m1_turn_churn_per_word"], s["m1_turn_churn_per_word_step"],
        ))

    out.executemany("""
        INSERT OR REPLACE INTO turn_metrics(
            run_id, puzzle_id, model, turn_id, turn_index,
            guess_status, feedback_returned, parse_success,
            solved_before_json, solved_after_json,
            n_candidates, n_kept, n_belief_updates,
            n_historical_guess_matches, n_solved_overlap_candidates,
            n_stripped_candidates, n_blocked_small_core, has_submit_anchor,
            q_start, q_first, q_pre_submit, q_submit, q_end, q_peak, q_regret,
            within_turn_gain, action_gain,
            context_reset_delta, next_turn_initial_gain, response_delta,
            m1_turn_churn_total, m1_turn_churn_per_word, m1_turn_churn_per_word_step
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, rows)


def write_run_metrics(out, *, run_id, puzzle_id, model, n_turns, metrics):
    cols = [
        "run_id", "puzzle_id", "model", "n_turns",
        "n_steps", "n_turn_starts", "n_candidates", "n_kept", "n_action_anchors",
        "n_belief_updates", "n_historical_guess_matches", "n_solved_overlap_candidates",
        "n_stripped_candidates", "n_blocked_small_core",
        "m4_regret", "m4_peak", "m4_final", "m4_peak_frac",
        "m4_peak_global_step", "m4_peak_turn_index",
        "m1_churn_total", "m1_churn_per_word", "m1_churn_per_word_step",
        "m1_churn_transitions", "m1_correct_abandonment", "m1_lockin_mean", "m1_lockin_json",
        "m2_blast_duration", "m2_wrong_word_steps", "m2_blocked_word_steps",
        "m2_blocked_blast_sum", "m2_mean_blast_radius", "m2_max_blast_radius",
        "m3_dstruct", "m3_word_violations", "m3_group_violations", "m3_size_violations",
        "final_exact_groups", "final_ari", "trajectory_solved",
        "db_solved", "db_binary_score", "guesses_used", "mistakes_used",
    ]

    values = [run_id, puzzle_id, model, n_turns] + [metrics[c] for c in cols[4:]]
    placeholders = ",".join("?" for _ in cols)
    out.execute(
        f"INSERT OR REPLACE INTO run_metrics({','.join(cols)}) VALUES ({placeholders})",
        values,
    )


# =============================================================================
# MAIN
# =============================================================================

def run(args):
    tier1 = sqlite3.connect(f"file:{args.tier1_db}?mode=ro", uri=True)
    tier2 = sqlite3.connect(f"file:{args.tier2_db}?mode=ro", uri=True)
    out = sqlite3.connect(args.out)
    out.execute("PRAGMA journal_mode=WAL;")
    ensure_out(out)

    source_connections = []
    for source_path in args.source_db:
        source_connections.append((
            source_path,
            sqlite3.connect(f"file:{source_path}?mode=ro", uri=True),
        ))

    started = time.time()
    scored = unchanged = recomputed = skipped = 0
    eligible_seen = 0
    stop = False

    for source_path, source in source_connections:
        if stop:
            break

        for row in source_runs(source, model_filter=args.model_filter):
            (
                run_id, puzzle_id, model, solved, binary_score,
                guesses_used, mistakes_used, ground_truth_json,
            ) = row

            if args.run_id and run_id != args.run_id:
                continue

            # Optional deterministic offset for smoke tests. Applied after
            # source/model/run-id filtering but before resume/signature logic.
            if eligible_seen < args.skip_runs:
                eligible_seen += 1
                continue
            eligible_seen += 1

            if args.limit_runs is not None and scored >= args.limit_runs:
                stop = True
                break

            signature = current_signature(
                source=source, tier1=tier1, tier2=tier2, run_id=run_id
            )
            previous_signature = get_processed_signature(out, run_id)

            if previous_signature == signature:
                exists = out.execute(
                    "SELECT 1 FROM run_metrics WHERE run_id = ?", (run_id,)
                ).fetchone()
                if exists:
                    unchanged += 1
                    continue

            if previous_signature is not None:
                clear_run_output(out, run_id)
                recomputed += 1

            out.execute("DELETE FROM skipped_runs WHERE run_id = ?", (run_id,))
            out.commit()

            gold = parse_gold_groups(ground_truth_json)
            if not gold:
                record_skip(out, run_id=run_id, source_db=source_path, reason="invalid ground truth")
                skipped += 1
                continue

            raw_turns = load_raw_turns(
                source, run_id=run_id, puzzle_id=puzzle_id, model=model
            )
            tier1_candidates = load_tier1_candidates(tier1, run_id)
            tier2_labels = load_tier2_labels(tier2, run_id)

            series, error = build_run_series(
                raw_turns=raw_turns,
                candidates_by_turn=tier1_candidates,
                labels_by_turn=tier2_labels,
                gold=gold,
            )

            if error:
                record_skip(out, run_id=run_id, source_db=source_path, reason=error)
                skipped += 1
                continue
            if not series:
                record_skip(out, run_id=run_id, source_db=source_path, reason="empty trajectory")
                skipped += 1
                continue

            source_run = {
                "solved": solved,
                "binary_score": binary_score,
                "guesses_used": guesses_used,
                "mistakes_used": mistakes_used,
            }
            metrics = compute_run_metrics(series=series, gold=gold, source_run=source_run)
            turn_summaries = compute_turn_metrics(series=series, raw_turns=raw_turns)

            if not args.no_step_cache:
                write_step_cache(
                    out, run_id=run_id, puzzle_id=puzzle_id, model=model, series=series
                )
            write_turn_metrics(out, turn_summaries)
            write_run_metrics(
                out,
                run_id=run_id,
                puzzle_id=puzzle_id,
                model=model,
                n_turns=len(raw_turns),
                metrics=metrics,
            )

            out.execute("""
                INSERT OR REPLACE INTO processed_runs(
                    run_id, source_db, raw_turn_count,
                    tier1_candidate_count, tier2_label_count,
                    policy_version, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, source_path, signature[0], signature[1], signature[2],
                POLICY_VERSION, time.strftime("%Y-%m-%dT%H:%M:%S"),
            ))
            out.execute("DELETE FROM skipped_runs WHERE run_id = ?", (run_id,))
            out.commit()

            scored += 1
            if scored % 50 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"{scored} processed this invocation | "
                    f"{unchanged} unchanged skipped | "
                    f"{recomputed} recomputed | "
                    f"{skipped} incomplete/skipped | "
                    f"{scored / elapsed:.2f} runs/s",
                    flush=True,
                )

    elapsed = time.time() - started
    print()
    print(
        "DONE\n"
        f"  policy        : {POLICY_VERSION}\n"
        f"  processed now : {scored}\n"
        f"  unchanged     : {unchanged}\n"
        f"  recomputed    : {recomputed}\n"
        f"  skipped       : {skipped}\n"
        f"  elapsed       : {elapsed:.1f}s"
    )

    tier1.close()
    tier2.close()
    for _, conn in source_connections:
        conn.close()
    out.close()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tier1-db", required=True)
    p.add_argument("--tier2-db", required=True)
    p.add_argument(
        "--source-db",
        action="append",
        required=True,
        help="Raw multiturn DB. Repeat once per model DB.",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--model-filter", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--limit-runs", type=int, default=None)
    p.add_argument(
        "--skip-runs", type=int, default=0,
        help="Skip the first N eligible source runs before processing (useful for a disjoint smoke slice).",
    )
    p.add_argument("--no-step-cache", action="store_true")
    return p.parse_args()


def main():
    run(parse_args())



if __name__ == "__main__":
    main()
