
import logging
from flask import Blueprint, request, jsonify, current_app

sailing_bp = Blueprint("sailing_club", __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _normalize_slots(raw):
    """
    Ensure each slot is a 2-int list [start, end] with start <= end and duration >= 0.
    Filters out invalid entries gracefully.
    """
    norm = []
    for it in raw or []:
        if (not isinstance(it, (list, tuple))) or len(it) != 2:
            continue
        a, b = it
        if not isinstance(a, int) or not isinstance(b, int):
            continue
        # If swapped, fix; if equal, keep (zero duration won't affect merges or boats)
        if a > b:
            a, b = b, a
        norm.append([a, b])
    return norm

def merge_slots(slots):
    """
    Merge overlapping OR touching intervals.
    Example: [5,8] and [8,10] => [5,10].
    Assumes slots are normalized.
    """
    if not slots:
        return []

    slots.sort(key=lambda x: (x[0], x[1]))
    merged = []
    cur_s, cur_e = slots[0]

    for s, e in slots[1:]:
        # Overlap or adjacency => merge when s <= cur_e
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    merged.append([cur_s, cur_e])
    return merged

def min_boats_needed(slots):
    """
    Line-sweep: treat intervals as [start, end] with reuse allowed at equal boundary.
    That is, an interval ending at t frees a boat for another starting at t.
    Implement by processing (time, type) with end before start at same time.
    """
    if not slots:
        return 0

    events = []
    for s, e in slots:
        # Ignore zero-length intervals for boats (no time occupied).
        if e > s:
            events.append((s, 1))   # start +1
            events.append((e, -1))  # end -1

    if not events:
        return 0

    # Sort ends before starts at the same timestamp
    # type_order: end (-1) -> 0, start (+1) -> 1
    events.sort(key=lambda t: (t[0], 0 if t[1] == -1 else 1))

    cur = 0
    peak = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak

@sailing_bp.route("/sailing-club/submission", methods=["POST"])
def sailing_submission():
    # Require JSON and schema with testCases
    if request.content_type is None or "application/json" not in request.content_type.lower():
        logger.error("Invalid Content-Type: Expected application/json")
        return jsonify({"error": "Content-Type must be application/json"}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "testCases" not in payload or not isinstance(payload["testCases"], list):
        logger.error("Invalid payload: expected { testCases: [...] }")
        return jsonify({"error": "Invalid payload: expected { testCases: [...] }"}), 400

    solutions = []
    for tc in payload["testCases"]:
        # Log each test case received
        tc_id = None
        slots_in = []
        try:
            tc_id = tc.get("id")
            slots_in = tc.get("input", [])
            logger.info(f"Processing TestCase ID: {tc_id} with input: {slots_in}")
        except Exception as e:
            logger.error(f"Error extracting test case data for {tc_id}: {e}")
            continue

        if tc_id is None:
            logger.warning(f"Skipping malformed test case (missing ID)")
            continue

        try:
            norm = _normalize_slots(slots_in)
            merged = merge_slots(norm)
            boats = min_boats_needed(norm)

            solutions.append({
                "id": tc_id,
                "sortedMergedSlots": merged,
                "minBoatsNeeded": boats
            })
            logger.info(f"TestCase {tc_id} processed successfully with merged slots: {merged} and boats: {boats}")
        except Exception as e:
            logger.error(f"Error processing TestCase {tc_id}: {e}")

    # Optional: log final count of solutions processed
    try:
        current_app.logger.info(f"Processed {len(solutions)} test cases")
    except Exception:
        pass

    return jsonify({"solutions": solutions}), 200
