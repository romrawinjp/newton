# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Imitation Learning with Robot Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples imitation_cartpole --num-worlds 100
#
###########################################################################

import warp as wp

import newton
import newton.examples

from tqdm import tqdm

# Define policy network 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_state=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, act_dim),
            nn.Tanh()  # Assuming actions are normalized between -1 and 1
        )

    def forward(self, x):
        return self.net(x)
    

class Example:
    def __init__(self, viewer, num_worlds=8):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.viewer = viewer

        # --- PHYSICS SETUP ---
        cartpole = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1
        cartpole.default_body_armature = 0.1
        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"), 
            enable_self_collisions=False,
            collapse_fixed_joints=True)
        
        # Start HANGING DOWN (0.0) - The hard start!
        cartpole.joint_q[-3:] = [0.0, 0.5, 0.0] 

        builder = newton.ModelBuilder()
        builder.replicate(cartpole, self.num_worlds, spacing=(1.0, 2.0, 0.0))
        
        # finalize model
        self.model = builder.finalize()
        
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        # self.solver = newton.solvers.SolverSemiImplicit(self.model, joint_attach_ke=1600.0, joint_attach_kd=20.0)
        # self.solver = newton.solvers.SolverFeatherstone(self.model)
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)

        # --- LEARNING SETUP ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = "collect"
        self.obs_buffer = []
        self.act_buffer = []
        
        # Policy
        self.policy = PolicyNetwork(obs_dim=6, act_dim=1).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01) # Higher LR for faster adaptation
        
        # Wait until we collect 30,000 "stable" frames
        self.target_stable_samples = 100000 
        self.collected_samples = 0
        self.stable_steps = torch.zeros(self.num_worlds, device=self.device)
        self.graph = None
    

    def get_expert_action(self, q, qd):
        """
        This function implements a simple PD controller to balance the cartpole
        in the downward position (hanging down). The controller computes the force
        to apply to the cart based on the current state (position and velocity) of
        the cart and pole.
        """
        # 0 = Cart, 1 = Pole1, 2 = Pole2
        cart_pos = q[:, 0]
        cart_vel = qd[:, 0]
        theta = q[:, 1]
        dtheta = qd[:, 1]

        
        # 1. CONFIGURATION 
        # Balance Gains (Downward)
        # kp_pole can be low because gravity naturally pulls it down.
        # kd_pole (Damping) must be sufficient to stop the pendulum from swinging forever.
        kp_cart, kd_cart = 20.0, 10.0
        kp_pole, kd_pole = 10.0, 5.0 
        
        # Setting maxinum force
        max_force = 100.0

        # 2. ERROR CALCULATION 
        # Target angle is 0.0 (Hanging down)
        # Normalize angle to range [-pi, pi] to handle wrap-around
        angle_err = (theta + torch.pi) % (2 * torch.pi) - torch.pi
        
        # 3. CONTROL LAW (PD Controller) 
        # Simple negative feedback:
        # - Push against cart position error (keep center)
        # - Push against cart velocity (dampen cart)
        # - Push against angle error (help gravity slightly)
        # - Push against angular velocity (stop the swinging!)
        
        force = (
            - kp_cart * cart_pos 
            - kd_cart * cart_vel 
            - kp_pole * angle_err 
            - kd_pole * dtheta
        )

        # --- 4. CLAMP ---
        force = torch.clamp(force, -max_force, max_force)
        
        return force.unsqueeze(1)
    
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def train(self):
        pbar = tqdm(range(args.training_steps), desc="Training Policy")
        print(f"\n Training on {self.collected_samples} Stable Samples...")
        X = torch.cat(self.obs_buffer, dim=0).to(self.device)
        Y = torch.cat(self.act_buffer, dim=0).to(self.device)
        print(f"Training Data Shape: Observations {X.shape}, Actions {Y.shape}")
        
        self.policy.train()
        for epoch in pbar:
            self.optimizer.zero_grad()
            pred = self.policy(X)
            loss = nn.MSELoss()(pred, Y)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.5f}"})
                
        print("Training Done. Switching to Policy Control")
        self.obs_buffer = [] 
        self.act_buffer = []

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # --- IL LOGIC ---
            q_flat = wp.to_torch(self.state_0.joint_q)
            qd_flat = wp.to_torch(self.state_0.joint_qd)
            
            num_dofs = self.model.joint_count // self.num_worlds
            q_view = q_flat.view(self.num_worlds, num_dofs)
            qd_view = qd_flat.view(self.num_worlds, num_dofs)
            self.q_dim = q_view.shape[1]
            
            # FULL OBSERVATION: Cart, Pole1, Pole2, and velocities (6 dims)
            obs = torch.cat([q_view, qd_view], dim=1).float()

            if self.mode == "collect":
                # A. Expert Action
                action_torch = self.get_expert_action(q_view, qd_view)
                
                # B. INSTANT STABILITY CHECK
                # dist_to_target need to close to 0.0
                target_angle = torch.pi
                
                # Angle Check:
                dist_to_target = torch.abs(q_view[:, 1] - target_angle)
                is_angle_ok = dist_to_target < 0.001
                
                # Cart Check:
                is_cart_ok = torch.abs(q_view[:, 0]) < 2.0
                
                # Is it stable RIGHT NOW?
                is_now_stable = is_angle_ok & is_cart_ok
                
                valid_obs = obs[is_now_stable]
                valid_act = action_torch[is_now_stable]
                
                if len(valid_obs) > 0 :
                    self.obs_buffer.append(valid_obs.cpu())
                    self.act_buffer.append(valid_act.cpu())
                    self.collected_samples += len(valid_obs)

            elif self.mode == "eval":
                with torch.no_grad():
                    action_torch = self.policy(obs)

            # Apply Action
            all_torques = torch.zeros_like(q_view)
            all_torques[:, 0] = action_torch.squeeze()
            wp_torques = wp.from_torch(all_torques.flatten().contiguous())
            if self.control.joint_f is not None:
                wp.copy(self.control.joint_f, wp_torques)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        # Trigger training when we have enough STABLE data
        if self.mode == "collect" and (self.collected_samples >= self.target_stable_samples) and self.sim_time > 5.0:
            print(f"Goal Reached! {self.collected_samples} stable frames collected.")
            self.train()
            self.mode = "eval"
            # Optional: Reset robots to Upright for the student?
            # Or let the student take over immediately (might crash if it was currently swinging)
        
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-worlds", 
        type=int, 
        default=100, 
        help="Total number of simulated worlds."
    )
    parser.add_argument(
        "--training-steps", 
        type=int, 
        default=10000, 
        help="Number of training steps for imitation learning."
    )
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)
    newton.examples.run(example, args)