# Security

## What to report privately

- **A scoring loophole:** a way for a submission to earn points without building what the task asks for, or to make the grader stop, hang or write a wrong reward.
- **A leak:** anything that reveals a reference room we have not released, or any way for the agent inside a task's container to reach the task's reference room or its answers.
- **An ordinary security problem** in this code, such as a submission file that makes the grader read or write outside its folders.

## How

Write to **contact@pranalab.ai**. Please do not open a public issue: a loophole that is public before it is closed can change other entrants' results.

Say what you found, the verifier version (`verifier_version` in a `reward.json`) and, if you can, attach the smallest submission that shows it. We aim to answer within five working days.

## What happens next

We confirm the report, fix it, and release a new verifier version. Results graded before the fix are regraded where the fix can change them. With your agreement we name you in the release notes.
