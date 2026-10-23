# Module 2b: Defect bank for the bank-size ablation

Paper Stages 2 and 3, with one addition: a person reviews every crop the VLM rejects. Notebook
[`notebooks/02b_defect_bank_ablation.ipynb`](../../notebooks/02b_defect_bank_ablation.ipynb).

## Flow

1. **DiffMask** recovers a mask for each normal and anomaly pair. The flags are the paper's frozen
   configuration (Step 2.1).
2. **Candidates.** Each recovered region of at least 500 px becomes a candidate (Step 2.2).
3. **Crop.** Each candidate is cut as a square, 1.8 times the region's long side, with its substrate
   kept (Step 2.3).
4. **Contrast gate.** The masked pixels must differ from the crop's own substrate (Step 3.2). The
   comparison with the normal image is recorded but does not decide.
5. **VLM check.** Qwen2.5-VL-7B judges each crop that passes the gate.
6. **Human review.** Each VLM rejection is shown with its crops and the VLM's reason. A person
   approves it into the bank or keeps it rejected. The decisions are logged.
7. **Bank.** Accepted patches are written with their soft masks and a manifest.

## Human review

| Step | What happens |
|---|---|
| VLM rejects a crop | The crop is queued for review. Its status is `VLM REJECT`. |
| Person approves it | The crop is banked, with status `BANKED (human approved)`. |
| Person keeps it rejected | It stays out of the bank, and the decision is logged. |

Decisions are saved to `review_log.csv`. A committed run can apply saved decisions without asking,
if that file is attached as an input.

**Deviation from the paper.** Step 3.1 describes VLM-2 as the validity check, with no human step.
Any crop approved by hand is a deviation, and the paper must say so.

## Not yet done

A larger VLM could reduce how many crops need human review. This has not been run or tested, and it
may not remove the review entirely. It needs more GPU memory than the two T4s provide for the 7B
model.

## Outputs

| Path | Contents |
|---|---|
| `defect_bank/<category>/` | Square RGB patches and their `_alpha.png` masks |
| `defect_bank/manifest.csv` | One row per banked patch, with the columns Stages 4 and 5 read |
| `mask_recovery/` | Normal, anomaly, and mask per pair, plus the candidate, review, and statistics logs |
