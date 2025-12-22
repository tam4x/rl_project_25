# Project Description
- Project about Continual Reinforcement Learning with Policy Distillation in MuJoCo Environments
- Using different strategies to distill knowledge in a supervised fashion
- store memory from teacher model with different methods
- use different architecture for student and teacher model
- papers used in this project:
    - arXiv:1907.05855
    - arXiv:1606.09282
    - arXiv:1511.06295

# TODO:
- Create the teacher training models and produce the memory for student
    - One Teacher for every MuJoCo Task (Walker, Pusher, Reacher, Humanoid, Hopper, Half Cheetah, Ant)
- Test out different learning algorithms from SB3
- After having a robust training pipeline for every env, start with the student training
- store memory from teacher (their policy output, features etc.)
    - way of storying depends on distillation method (via CLIP or other methods, latent space alignment, basic distillation methods maybe LD, PKD, CrossKD)
    - store action probabilites and observations from the teacher
- train the student with the memory created
    - different loss functions can be used

    