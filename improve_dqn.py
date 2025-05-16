def visualize_agent_trajectory(self, initial_windfield_name, seed=0, max_steps=200):
        """
        Visualizes the agent's trajectory on a specific windfield.
        
        Args:
            initial_windfield_name: Name of the initial windfield
            seed: Seed for reproducibility
            max_steps: Maximum number of steps to take
        """
        
        # Create the environment
        initial_windfield = get_initial_windfield(initial_windfield_name)
        env = SailingEnv(**get_initial_windfield(initial_windfield_name))
        env.render_mode = 'human'  # Enable rendering
        
        # Reset the environment
        observation, _ = env.reset(seed=seed)
        
        # Temporarily modify the agent to use the policy network
        original_act = self.agent.act
        
        def pytorch_act(observation):
            state = self.build_enhanced_state(observation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Deterministic mode for visualization
                return self.actor.get_action(state_tensor, deterministic=True)
        
        self.agent.act = pytorch_act
        
        try:
            # Run the agent and collect positions
            positions = [env.position.copy()]
            actions = []
            rewards = []
            
            done = False
            truncated = False
            total_reward = 0
            
            for _ in range(max_steps):
                # Take an action
                action = self.agent.act(observation)
                actions.append(action)
                
                # Execute the action
                observation, reward, done, truncated, _ = env.step(action)
                rewards.append(reward)
                total_reward += reward
                
                # Record the position
                positions.append(env.position.copy())
                
                if done or truncated:
                    break
            
            positions = np.array(positions)
            
            # Create the visualization
            plt.figure(figsize=(10, 8))
            
            # Plot the grid
            plt.xlim(0, self.agent.grid_size[0])
            plt.ylim(0, self.agent.grid_size[1])
            plt.grid(True)
            
            # Plot the trajectory
            plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
            plt.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
            plt.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
            
            # Plot the wind field (simplified)
            grid_x = np.linspace(0, self.agent.grid_size[0], 10)
            grid_y = np.linspace(0, self.agent.grid_size[1], 10)
            X, Y = np.meshgrid(grid_x, grid_y)
            
            # Get the wind at each point on the grid
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            
            for i in range(len(grid_x)):
                for j in range(len(grid_y)):
                    # Assuming we have access to the _get_wind_at_position method
                    wind = env._get_wind_at_position(np.array([X[j, i], Y[j, i]]))
                    U[j, i] = wind[0]
                    V[j, i] = wind[1]
            
            # Normalize the wind vectors for better visualization
            wind_magnitude = np.sqrt(U**2 + V**2)
            max_magnitude = np.max(wind_magnitude)
            U = U / max_magnitude
            V = V / max_magnitude
            
            plt.quiver(X, Y, U, V, scale=10, color='cyan', alpha=0.6, label='Wind')
            
            # Add title and legend
            success_status = "Success" if done and not truncated else "Failure"
            plt.title(f"Agent's Trajectory ({success_status}, Reward: {total_reward:.2f})")
            plt.legend()
            
            plt.show()
            
            return positions, actions, rewards
        
        finally:
            # Restore the original action method
            self.agent.act = original_act


def build_enhanced_state(self, observation):
        """
        Preprocess the raw observation to construct an enhanced state vector
        with derived and normalized sailing-related features.

        Args:
            observation (np.ndarray): Raw environment observation (e.g. position, velocity, wind)

        Returns:
            np.ndarray: Enhanced state vector with normalized and engineered features
        """
        # --- Extract raw components ---
        position = np.array(observation[0:2])
        velocity = np.array(observation[2:4])
        wind = np.array(observation[4:6])

        # --- Compute goal-related information ---
        goal_position = np.array([
            self.agent.grid_size[0] - 1,
            self.agent.grid_size[1] - 1
        ])
        goal_vector = goal_position - position
        distance_to_goal = np.linalg.norm(goal_vector)
        max_distance = np.linalg.norm(self.agent.grid_size)
        normalized_distance = min(1.0, distance_to_goal / (max_distance + 1e-10))

        # Normalized direction to the goal
        goal_direction = goal_vector / (distance_to_goal + 1e-10)

        # --- Determine movement and wind directions ---
        velocity_magnitude = np.linalg.norm(velocity)
        wind_magnitude = np.linalg.norm(wind)

        # Boat direction: based on velocity or fallback to goal direction
        boat_direction = (
            velocity / (velocity_magnitude + 1e-10)
            if velocity_magnitude > 0.001 else goal_direction
        )

        # Wind direction and inverse (wind comes *from* this direction)
        wind_direction = (
            wind / (wind_magnitude + 1e-10)
            if wind_magnitude > 0.001 else np.zeros(2)
        )
        wind_from_direction = -wind_direction

        # --- Compute relative angles (normalized between 0 and 1) ---
        wind_relative_angle = np.arccos(
            np.clip(np.dot(boat_direction, wind_from_direction), -1.0, 1.0)
        ) / np.pi

        goal_relative_angle = np.arccos(
            np.clip(np.dot(boat_direction, goal_direction), -1.0, 1.0)
        ) / np.pi

        # --- Normalize raw inputs ---
        normalized_position = position / np.array(self.agent.grid_size)
        normalized_velocity = velocity / (self.max_observed_velocity + 1e-10)
        normalized_wind = wind / (self.max_observed_wind + 1e-10)

        # --- Assemble final enhanced state vector ---
        enhanced_state = np.concatenate([
            normalized_position,         # (2,) Normalized position
            normalized_velocity,         # (2,) Normalized velocity
            normalized_wind,             # (2,) Normalized wind vector
            goal_direction,              # (2,) Direction to goal
            [wind_relative_angle],       # (1,) Angle between wind and boat direction
            [goal_relative_angle],       # (1,) Angle between goal and boat direction
            [normalized_distance]        # (1,) Normalized distance to goal
        ])

        return enhanced_state