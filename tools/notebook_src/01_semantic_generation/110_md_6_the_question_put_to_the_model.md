## 6. The question put to the model

Defines the system prompt and the question. Neither mentions a specific category.

The model is asked for four short fields as JSON: which item carries the defect, what the defect
is, where it sits, and a name for it. An unparseable reply, or one repeating a defect already used
in that category, is retried up to three times and then falls back to a config default.

Edit `SYS_VLM1` and `ASK_VLM1` in section 2 if you want different wording.
