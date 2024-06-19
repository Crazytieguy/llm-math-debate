# LLM Math Debate

How good is training via Debate at incentivizing honesty? Will debate hold up against models that are initially fine tuned to trick the judge, or are reinforced more for winning with a wrong solution than winning with a correct solution?

This is my mostly abandoned [MATS 5.0](https://www.matsprogram.org/) project under the mentorship of David Rein and Julian Michael of the NYU Alignment Research Group. It was abandoned because open source models were demonstrably not smart enough to interface with the level of math in the AMPS dataset.

## Current state

- [x] Find a dataset of step by step solutions to math problems. The solutions should be simple enough for fine=tuned open source models to at least classify correctly. Dataset chosen - the relevant subset of [AMPS](https://github.com/hendrycks/math).
- [x] Filter and process AMPS to exclude mistakes and format the data nicer.
- [x] Write the code and prompts necessary for inserting errors into correct solutions with GPT-4.
- [ ] Find an open source model that when fine tuned as a solution classifier gets decent but not great accuracy. This will be the judge.
- [ ] Find an open source model that gets significantly better accuracy than the judge model. This will be the debater.
- [ ] Fine-tune the debater and judge on the debate format (details TBD).
- [ ] Train the debater via debate - the result will be the non-adversarial baseline.
- [ ] Try various adversarial initializations and set ups and see how they affect the final model's behavior and accuracy.
