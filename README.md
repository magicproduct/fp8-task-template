# FP8 stochastic rounding

- Write a python function that converts a torch tensor of type {bfloat16}
  to [fp8](https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support) with N mantissa bits (N is an argument to
  this function).
- The fp8 tensor should be stored as a uint8 tensor because not all GPUs support fp8 natively. Cast to fp8, not to
  uint8. Clarifying Note: This task is about shifting bits, not using .to(torch.uint8).
- Also, write a function to convert the int8-based tensor back to bfloat16
- Your function should stochastically round (https://nhigham.com/2020/07/07/what-is-stochastic-rounding/) the source
  tensor. Note that there are edge cases, all of which should be considered.
- Write unit test that assert the expected value of the casting function is close to the source tensor to validate
  your stochastic rounding implementation. The unit test is part of the task, and we're looking for detailed and thorough assertions
  that target the key functionality precisely.

## Environment
- Use a py39 environment
- Ensure that your task can be executed on an x86 CPU
- Start from `src/solution.py`

## FAQ

#### Can I use numpy?
No, only pytorch for the solution. You may use numpy for your unit tests.

#### What is the "scaling_factor" argument supposed to mean?
In fp8, you usually use per-tensor scaling factors. You get in some floats, the max of that tensor can be, e.g., 5, and then you want to convert to fp8. But you don’t want to waste precious precision space, so you scale the tensor up by max_value_fp8_type_supports/tensor.max() – that’s the scaling_factor.

#### Do I have to consider edge cases?
Yes, your solution should work for NaN, inf and subnormals

#### For which fp8 datatypes should my solution work?
E4M3 and E5M2

#### How long should this take me?
Successful candidates usually take between a few hours and a full day to arrive at a complete solution

#### How do I submit the task
Invite @magic-screening-tasks-gh-sa and @EricSteinberger to your _private_ github repo and let us know you are ready.
