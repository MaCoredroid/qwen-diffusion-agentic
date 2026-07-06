## Qwen Code invocation prompt (container runtime)

Read the task prompt at ./AGENTS.md and complete it in this workspace. Edit the source files directly to implement the fix. Do not write a diff file -- modify the files in place so that running the project's tests passes the tests described in the prompt. Do NOT modify any test files.

## AGENTS.md (workspace/conan-io__conan-10213)

# SWE-Bench task: conan-io__conan-10213

**Repo:** `conan-io/conan`  
**Base commit:** `106e54c96118a1345899317bf3bebaf393c41574`  
**Version:** `1.44`  

## Problem statement

[feature] Improve custom CMakeToolchain blocks extension
From https://github.com/conan-io/docs/pull/2312, lets avoid the ``import CMakeToolchainBlock`` altogether, using Python dynamism.


## Required behavior

Implement the fix described in the problem statement by editing the source files in this workspace. Do NOT modify any test files. The hidden grader will apply its own test patch and run the test suite; your code must make those tests pass without breaking existing ones.

## How to work (important)

- Reason carefully and thoroughly before each tool call. First inspect the relevant source files to confirm your understanding of the bug, then make the minimal correct edit.
- Do NOT spend your time trying to `pip install` or build/conda the project -- the grader runs in its own prepared environment. If an install/build command fails, do not retry it; just edit the source.
- You MUST finish by leaving an actual code change in the working tree. Do not stop until you have edited the source files to implement the fix.

