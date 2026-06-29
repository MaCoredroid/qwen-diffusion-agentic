#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT = ROOT / "data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl"
DEFAULT_SYSTEM = "You are a helpful assistant."


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


VOICE_TOOLS = [
    tool(
        "activate_voice_command",
        "Activates a device using a spoken voice command.",
        {
            "command": {"type": "string", "description": "The voice command to be executed."},
            "device_type": {
                "type": "string",
                "description": "The type of device to be controlled.",
                "enum": ["light", "thermostat", "camera"],
            },
            "location": {"type": "string", "description": "The location of the device within the home."},
        },
        ["command", "device_type", "location"],
    ),
    tool(
        "set_thermostat",
        "Sets the temperature of the smart thermostat to a specified value.",
        {
            "temperature": {"type": "number", "description": "The desired temperature to set on the thermostat."},
            "location": {"type": "string", "description": "The location of the thermostat within the home."},
        },
        ["temperature", "location"],
    ),
    tool(
        "activate_security_cameras",
        "Changes the status of security cameras to on or off.",
        {
            "status": {"type": "string", "description": "The desired status of the security cameras.", "enum": ["on", "off"]},
            "mode": {"type": "string", "description": "The mode to set for the security cameras.", "enum": ["home", "away", "night"]},
        },
        ["status", "mode"],
    ),
]


SECURITY_TOOLS = [
    tool(
        "install_smart_lock",
        "Installs a new smart lock on a specified door using the provided model details and installation code.",
        {
            "door": {"type": "string", "description": "The door where the smart lock will be installed."},
            "model": {"type": "string", "description": "The model of the smart lock to be installed."},
            "installation_code": {"type": "string", "description": "The installation code required to set up the smart lock."},
        },
        ["door", "model", "installation_code"],
    ),
    tool(
        "configure_motion_detectors",
        "Configures motion detectors in a specified location using the provided model details and installation code.",
        {
            "location": {"type": "string", "description": "The location where the motion detectors will be configured."},
            "model": {"type": "string", "description": "The model of the motion detectors to be configured."},
            "installation_code": {"type": "string", "description": "The installation code required to set up the motion detectors."},
        },
        ["location", "model", "installation_code"],
    ),
    tool(
        "activate_security_alarm",
        "Activates the security alarm system using the provided system activation code.",
        {
            "system_code": {"type": "string", "description": "The activation code for the security alarm system."},
        },
        ["system_code"],
    ),
]


def compact_call(name, arguments):
    payload = {"name": name, "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def compact_calls(calls):
    return "\n".join(compact_call(call["name"], call["arguments"]) for call in calls)


def available_tool_names(tools):
    names = []
    for item in tools:
        fn = item.get("function", item)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.append(name)
    return names


def eval_row(row_id, family, task, user, tools, calls, bad_draft_calls=None):
    gold = compact_calls(calls)
    parsed_calls, invalid = extract_tool_calls(gold)
    bad_draft = compact_calls(bad_draft_calls) if bad_draft_calls else ""
    return {
        "source": "synthetic_multicall_failure_analogue",
        "id": row_id,
        "analogue_family": family,
        "category": "IoT and Home Automation",
        "task": task,
        "tools": tools,
        "prompt_messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": user.strip()},
        ],
        "gold_assistant": gold,
        "gold_tool_names": [call["name"] for call in calls],
        "available_tool_names": available_tool_names(tools),
        "gold_tool_calls": parsed_calls,
        "gold_invalid_tool_json_count": invalid,
        "assistant": "",
        "contextual_projection_assistant": "",
        "bad_draft_assistant": bad_draft,
    }


def voice_cases():
    rows = []
    rows.append(
        eval_row(
            "synthetic_voice_command_camera_001",
            "voice_command_camera",
            "Perform home tasks using spoken commands",
            """
I am unloading supplies and need the home assistant to execute the spoken commands instead of using the camera status shortcut directly.

Please call the needed functions for this list:
1. Voice command for pantry lights - command: "Switch on pantry lights"; device_type: "light"; location: "pantry".
2. Thermostat adjustment - temperature: 68; location: "upstairs hallway".
3. Voice command for driveway cameras - command: "Start driveway cameras in travel watch"; device_type: "camera"; location: "driveway".
""",
            VOICE_TOOLS,
            [
                {"name": "activate_voice_command", "arguments": {"command": "Switch on pantry lights", "device_type": "light", "location": "pantry"}},
                {"name": "set_thermostat", "arguments": {"temperature": 68, "location": "upstairs hallway"}},
                {
                    "name": "activate_voice_command",
                    "arguments": {"command": "Start driveway cameras in travel watch", "device_type": "camera", "location": "driveway"},
                },
            ],
            [
                {"name": "activate_voice_command", "arguments": {"command": "Switch on pantry lights", "device_type": "light", "location": "pantry"}},
                {"name": "set_thermostat", "arguments": {"temperature": 68, "location": "upstairs hallway"}},
                {"name": "activate_security_cameras", "arguments": {"status": "on", "mode": "away"}},
            ],
        )
    )
    rows.append(
        eval_row(
            "synthetic_voice_command_camera_002",
            "voice_command_camera",
            "Use voice commands for mixed smart-home setup",
            """
The wall panel is out of reach, so use voice-command actions for the spoken requests. The camera item is intentionally a spoken command, not the direct security-camera API.

Please execute:
1. Voice command for porch lights - command: "Turn porch lights to evening mode"; device_type: "light"; location: "front porch".
2. Thermostat adjustment - temperature: 71; location: "family room".
3. Voice command for garage cameras - command: "Arm garage cameras for night watch"; device_type: "camera"; location: "garage".
""",
            VOICE_TOOLS,
            [
                {
                    "name": "activate_voice_command",
                    "arguments": {"command": "Turn porch lights to evening mode", "device_type": "light", "location": "front porch"},
                },
                {"name": "set_thermostat", "arguments": {"temperature": 71, "location": "family room"}},
                {
                    "name": "activate_voice_command",
                    "arguments": {"command": "Arm garage cameras for night watch", "device_type": "camera", "location": "garage"},
                },
            ],
            [
                {
                    "name": "activate_voice_command",
                    "arguments": {"command": "Turn porch lights to evening mode", "device_type": "light", "location": "front porch"},
                },
                {"name": "set_thermostat", "arguments": {"temperature": 71, "location": "family room"}},
                {"name": "activate_security_cameras", "arguments": {"status": "on", "mode": "night"}},
            ],
        )
    )
    rows.append(
        eval_row(
            "synthetic_voice_command_camera_003",
            "voice_command_camera",
            "Issue spoken commands while preparing to leave",
            """
I am stepping out with wet paint on my hands, so I only want the spoken-command route used for the voice items.

Call the functions for:
1. Voice command for studio lights - command: "Dim studio lights to half"; device_type: "light"; location: "studio".
2. Thermostat adjustment - temperature: 66; location: "bedroom".
3. Voice command for side-yard cameras - command: "Enable side-yard cameras for away check"; device_type: "camera"; location: "side yard".
""",
            VOICE_TOOLS,
            [
                {"name": "activate_voice_command", "arguments": {"command": "Dim studio lights to half", "device_type": "light", "location": "studio"}},
                {"name": "set_thermostat", "arguments": {"temperature": 66, "location": "bedroom"}},
                {
                    "name": "activate_voice_command",
                    "arguments": {"command": "Enable side-yard cameras for away check", "device_type": "camera", "location": "side yard"},
                },
            ],
            [
                {"name": "activate_voice_command", "arguments": {"command": "Dim studio lights to half", "device_type": "light", "location": "studio"}},
                {"name": "set_thermostat", "arguments": {"temperature": 66, "location": "bedroom"}},
                {"name": "activate_security_cameras", "arguments": {"status": "on", "mode": "away"}},
            ],
        )
    )
    rows.append(
        eval_row(
            "synthetic_voice_command_camera_004",
            "voice_command_camera",
            "Run smart-home voice commands before a trip",
            """
Before leaving for the station, I need three home changes made. Use the voice-command tool for the spoken commands even when the spoken command mentions cameras.

Tasks:
1. Voice command for mudroom lights - command: "Bring up mudroom lights"; device_type: "light"; location: "mudroom".
2. Thermostat adjustment - temperature: 69; location: "main floor".
3. Voice command for backyard cameras - command: "Set backyard cameras to trip watch"; device_type: "camera"; location: "backyard".
""",
            VOICE_TOOLS,
            [
                {"name": "activate_voice_command", "arguments": {"command": "Bring up mudroom lights", "device_type": "light", "location": "mudroom"}},
                {"name": "set_thermostat", "arguments": {"temperature": 69, "location": "main floor"}},
                {
                    "name": "activate_voice_command",
                    "arguments": {"command": "Set backyard cameras to trip watch", "device_type": "camera", "location": "backyard"},
                },
            ],
            [
                {"name": "activate_voice_command", "arguments": {"command": "Bring up mudroom lights", "device_type": "light", "location": "mudroom"}},
                {"name": "set_thermostat", "arguments": {"temperature": 69, "location": "main floor"}},
                {"name": "activate_security_cameras", "arguments": {"status": "on", "mode": "away"}},
            ],
        )
    )
    return rows


def security_cases():
    rows = []
    rows.append(
        eval_row(
            "synthetic_security_codes_001",
            "security_installation_codes",
            "Set up smart-home security devices with separate codes",
            """
Please set up these security components. Each device has its own code, so keep the lock, motion-detector, and alarm codes separated.

1. Install smart lock - door: "side entry"; model: "Orbit Secure 8"; installation_code: "OR8-441L".
2. Configure motion detectors - location: "basement"; model: "Bosch TriTech 2000"; installation_code: "BT2K-77M".
3. Activate security alarm - system_code: "ARM-5520".
""",
            SECURITY_TOOLS,
            [
                {"name": "install_smart_lock", "arguments": {"door": "side entry", "model": "Orbit Secure 8", "installation_code": "OR8-441L"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "basement", "model": "Bosch TriTech 2000", "installation_code": "BT2K-77M"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "ARM-5520"}},
            ],
            [
                {"name": "install_smart_lock", "arguments": {"door": "side entry", "model": "Orbit Secure 8", "installation_code": "OR8-441L"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "basement", "model": "Bosch TriTech 2000", "installation_code": "OR8-441L"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "ARM-5520"}},
            ],
        )
    )
    rows.append(
        eval_row(
            "synthetic_security_codes_002",
            "security_installation_codes",
            "Upgrade locks, detectors, and alarm service",
            """
I am updating several security devices. Do not reuse one code for another device.

1. Install smart lock - door: "back patio door"; model: "Schlage Encode Plus"; installation_code: "SEP-6042".
2. Configure motion detectors - location: "upper hallway"; model: "Ecolink PIR Pro"; installation_code: "ECO-31PIR".
3. Activate security alarm - system_code: "SAFE-904B".
""",
            SECURITY_TOOLS,
            [
                {"name": "install_smart_lock", "arguments": {"door": "back patio door", "model": "Schlage Encode Plus", "installation_code": "SEP-6042"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "upper hallway", "model": "Ecolink PIR Pro", "installation_code": "ECO-31PIR"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "SAFE-904B"}},
            ],
            [
                {"name": "install_smart_lock", "arguments": {"door": "back patio door", "model": "Schlage Encode Plus", "installation_code": "SEP-6042"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "upper hallway", "model": "Ecolink PIR Pro", "installation_code": "SEP-6042"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "SAFE-904B"}},
            ],
        )
    )
    rows.append(
        eval_row(
            "synthetic_security_codes_003",
            "security_installation_codes",
            "Configure a new security stack",
            """
Please complete this security setup in order. The detector installation code is different from the lock code.

1. Install smart lock - door: "garage service door"; model: "Aqara U100"; installation_code: "AQ-U1-552".
2. Configure motion detectors - location: "home office"; model: "Nest Detect Pro"; installation_code: "NDP-19M5".
3. Activate security alarm - system_code: "ALR-731Q".
""",
            SECURITY_TOOLS,
            [
                {"name": "install_smart_lock", "arguments": {"door": "garage service door", "model": "Aqara U100", "installation_code": "AQ-U1-552"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "home office", "model": "Nest Detect Pro", "installation_code": "NDP-19M5"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "ALR-731Q"}},
            ],
            [
                {"name": "install_smart_lock", "arguments": {"door": "garage service door", "model": "Aqara U100", "installation_code": "AQ-U1-552"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "home office", "model": "Nest Detect Pro", "installation_code": "AQ-U1-552"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "ALR-731Q"}},
            ],
        )
    )
    rows.append(
        eval_row(
            "synthetic_security_codes_004",
            "security_installation_codes",
            "Install and activate home security hardware",
            """
I need the following three security actions performed. The installation code nearest each device is the one to use for that device.

1. Install smart lock - door: "front gate"; model: "Level Lock Touch"; installation_code: "LLT-220G".
2. Configure motion detectors - location: "guest suite"; model: "Ring Motion Zone"; installation_code: "RMZ-842D".
3. Activate security alarm - system_code: "GUARD-618".
""",
            SECURITY_TOOLS,
            [
                {"name": "install_smart_lock", "arguments": {"door": "front gate", "model": "Level Lock Touch", "installation_code": "LLT-220G"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "guest suite", "model": "Ring Motion Zone", "installation_code": "RMZ-842D"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "GUARD-618"}},
            ],
            [
                {"name": "install_smart_lock", "arguments": {"door": "front gate", "model": "Level Lock Touch", "installation_code": "LLT-220G"}},
                {
                    "name": "configure_motion_detectors",
                    "arguments": {"location": "guest suite", "model": "Ring Motion Zone", "installation_code": "LLT-220G"},
                },
                {"name": "activate_security_alarm", "arguments": {"system_code": "GUARD-618"}},
            ],
        )
    )
    return rows


def build_rows():
    return voice_cases() + security_cases()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = build_rows()
    write_jsonl(args.out_jsonl, rows)
    summary = {
        "out_jsonl": str(args.out_jsonl),
        "rows": len(rows),
        "source": "synthetic_multicall_failure_analogue",
        "family_counts": dict(sorted(Counter(row["analogue_family"] for row in rows).items())),
        "ids": [row["id"] for row in rows],
        "gold_tool_call_count": sum(len(row["gold_tool_names"]) for row in rows),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
