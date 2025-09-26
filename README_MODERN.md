# NoReward-RL: Enhanced Curiosity-Driven Exploration

This is an enhanced and modernized version of Deepak Pathak's "noreward-rl" repository, implementing curiosity-driven exploration for reinforcement learning with significant improvements and modern tooling.

## 🚀 Key Enhancements

### ✅ **Modern Environment Support**
- **Gymnasium Integration**: Full compatibility with modern `gymnasium` (successor to `gym`)
- **Backward Compatibility**: Maintains support for legacy environments
- **Enhanced Wrappers**: Improved environment preprocessing and observation handling

### ✅ **Advanced Logging & Monitoring**
- **Weights & Biases Integration**: Comprehensive experiment tracking and visualization
- **TensorBoard Support**: Traditional logging with configurable options
- **Curiosity Metrics**: Specialized logging for intrinsic motivation and exploration

### ✅ **Enhanced Demo & Evaluation**
- **Video Recording**: Automatic MP4/GIF generation using `gymnasium.RecordVideo`
- **Clean Evaluation Scripts**: Modular evaluation and recording tools
- **Performance Reports**: Detailed statistics and visualizations

### ✅ **Hard Exploration Benchmarks**
- **Atari Hard Exploration**: Support for Montezuma's Revenge, Pitfall, and other challenging games
- **Standardized Benchmarks**: Consistent evaluation across different environments
- **Comprehensive Reporting**: Detailed performance analysis and comparisons

### ✅ **Production-Ready Code**
- **Type Hints**: Full type annotation for better IDE support and maintainability
- **Modular Design**: Clean separation of concerns and easy extensibility
- **Comprehensive Documentation**: Detailed docstrings and usage examples
- **Error Handling**: Robust error handling and validation

## 📦 Installation

### Prerequisites
- **Python 3.8-3.11** (tested on all versions)
- **TensorFlow 2.16+** (CPU/GPU support)
- **CUDA** (optional, for GPU acceleration)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/jazhanma/noreward-rl.git
cd noreward-rl

# Install core dependencies (pinned versions for reproducibility)
pip install -r requirements.txt

# Install in development mode with all features
pip install -e ".[dev,all]"
```

### Environment-Specific Dependencies
```bash
# For VizDoom environments
pip install vizdoom>=1.2.0

# For Super Mario Bros
pip install gym-super-mario-bros>=7.4.0

# For Atari games (included in main requirements)
# pip install gymnasium[atari]==1.0.0

# For Weights & Biases logging (included in main requirements)
# pip install wandb==0.16.6
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
flake8 src/ tests/
mypy src/
```

### Reproducible Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install exact versions for reproducibility
pip install -r requirements.txt

# Verify installation
python -c "import gymnasium, tensorflow, wandb; print('✅ All dependencies installed successfully!')"
```

## 🎮 Supported Environments

### **VizDoom (First-Person Shooter)**
- **My Way Home**: Navigate to target location
- **Reward Sparsity Levels**: Dense, Sparse, Very Sparse
- **Custom Scenarios**: Easy configuration for new levels

### **Super Mario Bros**
- **Platformer Game**: Classic side-scrolling action
- **Distance Rewards**: Progress-based motivation
- **Fast Resets**: Optimized for training efficiency

### **Atari Games**
- **Standard Games**: Pong, Breakout, Space Invaders, etc.
- **Hard Exploration**: Montezuma's Revenge, Pitfall, Private Eye, etc.
- **Preprocessing**: Frame stacking, resizing, and normalization

## 🚀 Quick Start

### **Training a Curiosity-Driven Agent**

```bash
# Train on Doom with Weights & Biases logging
python src/train_modern.py --env-id doom --use-wandb --experiment-name "doom-curiosity"

# Train on Mario without external rewards
python src/train_modern.py --env-id mario --no-reward --unsup action --use-wandb

# Train on hard exploration Atari game
python src/train_modern.py --env-id MontezumaRevenge-v5 --unsup action --num-workers 8
```

### **Evaluating Trained Models**

```bash
# Evaluate and record videos
python scripts/eval_and_record.py --env-id doom --model-path models/doom_ICM --record --num-episodes 10

# Run greedy evaluation
python scripts/eval_and_record.py --env-id mario --model-path models/mario_ICM --greedy

# Save individual frames
python scripts/eval_and_record.py --env-id MontezumaRevenge-v5 --model-path models/montezuma_ICM --save-frames
```

### **Benchmarking Hard Exploration Games**

```bash
# Benchmark a single game
python scripts/benchmark_hard_exploration.py --env-id MontezumaRevenge-v5 --model-path models/montezuma_ICM

# Benchmark all hard exploration games
python scripts/benchmark_hard_exploration.py --all-games --model-path models/atari_ICM --num-episodes 20

# Generate comprehensive report
python scripts/benchmark_hard_exploration.py --all-games --model-path models/atari_ICM --use-wandb
```

## 🧠 Core Algorithms

### **Intrinsic Curiosity Module (ICM)**
The ICM consists of two neural networks:

1. **Forward Model**: Predicts next state given current state and action
   - Loss: `||f(φ(s_t), a_t) - φ(s_{t+1})||²`
   - Higher prediction error = more curiosity = more exploration

2. **Inverse Model**: Predicts action taken given two consecutive states
   - Loss: Cross-entropy between predicted and actual action
   - Helps learn useful state representations

### **A3C (Asynchronous Advantage Actor-Critic)**
- **Distributed Training**: Multiple workers with shared parameters
- **Actor-Critic Architecture**: Separate policy and value networks
- **LSTM Memory**: Recurrent connections for temporal dependencies

## 📊 Logging & Monitoring

### **Weights & Biases Integration**
```python
# Automatic logging of:
# - Training losses (policy, value, entropy)
# - Curiosity metrics (intrinsic rewards, prediction errors)
# - Episode statistics (rewards, lengths, times)
# - Environment-specific metrics (distance, position)
# - Model parameters and gradients
```

### **TensorBoard Support**
```bash
# View training progress
tensorboard --logdir logs/

# Compare different experiments
tensorboard --logdir logs/ --port 6006
```

## 🔧 Configuration

### **Environment-Specific Settings**
```python
# Automatically configured based on environment
ENV_CONFIGS = {
    'doom': {
        'PREDICTION_BETA': 0.01,
        'ENTROPY_BETA': 0.01,
        'FRAME_SKIP': 4,
    },
    'mario': {
        'PREDICTION_BETA': 0.2,
        'ENTROPY_BETA': 0.0005,
        'FRAME_SKIP': 6,
    },
    'montezuma': {
        'PREDICTION_BETA': 0.1,
        'ENTROPY_BETA': 0.01,
        'FRAME_SKIP': 4,
    },
}
```

### **Command Line Options**
```bash
# Training options
--env-id              # Environment identifier
--num-workers         # Number of worker processes
--max-steps           # Maximum training steps
--learning-rate       # Learning rate
--prediction-beta     # Curiosity weight
--unsup              # Unsupervised learning type

# Logging options
--use-wandb           # Enable Weights & Biases
--use-tensorboard     # Enable TensorBoard
--experiment-name     # Experiment identifier

# Evaluation options
--num-episodes        # Number of evaluation episodes
--render              # Visualize environment
--record              # Record videos
--greedy              # Use greedy policy
```

## 📈 Performance Tips

### **Training Optimization**
1. **Use Multiple Workers**: Scale training with `--num-workers`
2. **Environment-Specific Settings**: Let the system auto-configure hyperparameters
3. **Monitor Logs**: Use Weights & Biases for real-time monitoring
4. **Hardware**: Use GPU acceleration when available

### **Evaluation Best Practices**
1. **Use Greedy Policy**: For consistent evaluation results
2. **Multiple Episodes**: Run at least 10 episodes for reliable statistics
3. **Record Videos**: Visual inspection helps understand agent behavior
4. **Save Frames**: For detailed analysis of agent decisions

## 🔬 Research Applications

### **Sparse Reward Environments**
- **Robotics**: Manipulation tasks with sparse success signals
- **Navigation**: Exploration in unknown environments
- **Game Playing**: Learning complex strategies without explicit rewards

### **Unsupervised Learning**
- **Representation Learning**: Learning useful state representations
- **Exploration**: Discovering new strategies and behaviors
- **Transfer Learning**: Pre-training for downstream tasks

## 📚 Citation

If you use this enhanced version in your research, please cite both the original paper and this implementation:

```bibtex
@inproceedings{pathakICMl17curiosity,
    Author = {Pathak, Deepak and Agrawal, Pulkit and
              Efros, Alexei A. and Darrell, Trevor},
    Title = {Curiosity-driven Exploration by Self-supervised Prediction},
    Booktitle = {International Conference on Machine Learning ({ICML})},
    Year = {2017}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ scripts/
isort src/ scripts/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original Authors**: Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell
- **OpenAI Universe**: For the original A3C implementation
- **Gymnasium Team**: For the modern environment interface
- **Weights & Biases**: For excellent experiment tracking tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/noreward-rl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/noreward-rl/discussions)
- **Documentation**: [Full Documentation](https://noreward-rl.readthedocs.io/)

---

**Happy Exploring! 🎯**

