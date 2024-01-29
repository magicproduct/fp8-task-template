# FP8 stochastic rounding

- Write a python function that converts a torch tensor of type {fp32, fp16 or bfloat16}
  to [fp8](https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support) with N mantissa bits (N is an argument to
  this function).
- The fp8 tensor should be stored as a uint8 tensor because not all GPUs support fp8 natively. Cast to fp8, not to
  uint8. Clarifying Note: This task is about shifting bits, not using .to(torch.uint8).
- Also write a function to convert the int8-based tensor back to fp16
- Your function should stochastically round (https://nhigham.com/2020/07/07/what-is-stochastic-rounding/) the source
  tensor. Note that there are edge cases, all of which should be considered.
- Write unit test thats assert the expected value of the casting function is close to the source tensor to validate
  your stochastic rounding implementation. The unit test is part of the task and we're looking for detailed and thorough assertions
  that target the key functionality precisely.

## Environment
- Use a py39 environment
- Ensure that your task can be executed on an x86 CPU
- Start from `src/solution.py`

## FAQ

#### Can I use numpy?
No, only pytorch

#### How long should this take me?
Successful candidates usually take between a few hours and a full day to arrive at a complete solution

#### How do I submit the task
Invite @magic-screening-tasks-gh-sa and @EricSteinberger to your _private_ github repo and let us know you are ready.
