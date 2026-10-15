# Fixed-output models

`combined_ep333.yaml`, `JC.yaml`, and `SK.yaml` preserve the original config
bytes. The inference adapter overrides batch size to 4, matching the research
backend. The sixth output is `other`, `lead_vocal`, or `baritone`, respectively;
these labels must not be treated as interchangeable.
