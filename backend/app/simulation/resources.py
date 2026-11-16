"""Known resource identities/units survive settlement; unknown is not zero."""
import math


def validate_resources(before: dict, after: dict, authoritative: dict | None = None):
    resources = before.get("actor_resources", {})
    current = after.get("actor_resources", {})
    if not resources:
        return
    if not isinstance(current, dict) or set(current) != set(resources):
        raise ValueError("Settlement must preserve configured resource actors")
    for actor, entries in resources.items():
        if not isinstance(entries, dict) or not isinstance(current[actor], dict) or set(current[actor]) != set(entries):
            raise ValueError("Settlement must preserve resource IDs")
        for identifier, previous in entries.items():
            value = current[actor][identifier]
            if not isinstance(previous, dict):
                raise ValueError("Resource records require value and unit fields")
            if not isinstance(value, dict) or value.get("unit") != previous.get("unit") or "value" not in value:
                raise ValueError("Settlement must preserve resource units and explicit values")
            amount = value["value"]
            path = f"actor_resources.{actor}.{identifier}.value"
            if previous.get("value") is None and amount is not None and path not in (authoritative or {}):
                raise ValueError("Unknown resource quantity cannot become a model-invented number; configure an explicit mechanism")
            if amount is not None and (isinstance(amount, bool) or not isinstance(amount, (int,float)) or not math.isfinite(amount)):
                raise ValueError("Resource quantity must be finite or unknown (null)")
