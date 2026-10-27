package shell_guard

import rego.v1

# A command may run without asking only if it is neither destructive nor networked,
# and the agent is working inside its own project folder.
default allow := false

allow if input.answer.action != "run"

allow if {
	input.answer.action == "run"
	not input.answer.destructive
	not input.answer.network
	startswith(input.facts.cwd, input.facts.project_root)
}

why_not contains "destructive commands always need a person" if {
	input.answer.action == "run"
	input.answer.destructive
}

why_not contains "network commands always need a person" if {
	input.answer.action == "run"
	input.answer.network
}

why_not contains "the agent is outside its project folder" if {
	input.answer.action == "run"
	not startswith(input.facts.cwd, input.facts.project_root)
}
