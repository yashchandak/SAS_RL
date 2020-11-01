import argparse
from datetime import datetime


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Parameters for Hyper-param sweep
        parser.add_argument("--base", default=-2, help="Base counter for Hyper-param search", type=int)
        parser.add_argument("--inc", default=-2, help="Increment counter for Hyper-param search", type=int)
        parser.add_argument("--hyper", default=-2, help="Which Hyper param settings", type=int)
        parser.add_argument("--seed", default=12345, help="seed for variance testing", type=int)

        # General parameters
        parser.add_argument("--save_count", default=10, help="Number of ckpts for saving results and model", type=int)
        parser.add_argument("--optim", default='sgd', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",
                            choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=False, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--restore", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model ckpts")
        parser.add_argument("--summary", default=True, type=self.str2bool,
                            help="--UNUSED-- Visual summary of various stats")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)

        # Book-keeping parameters
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")
        parser.add_argument("--experiment", default='Test_run', help="Name of the experiment")

        self.Env_n_Agent_args(parser)  # Decide the Environment and the Agent
        self.Main_AC_args(parser)  # General Basis, Policy, Critic
        self.SAS_PG(parser)  # Settings for stochastic action set

        self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser

    def Env_n_Agent_args(self, parser):
        # parser.add_argument("--algo_name", default='SAS_PGQV', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='SAS_NAC', help="Learning algorithm")
        parser.add_argument("--algo_name", default='SAS_Q', help="Learning algorithm")
        # parser.add_argument("--env_name", default='SAS_Reco', help="Environment to run the code")
        # parser.add_argument("--env_name", default='SF_Map', help="Environment to run the code")
        parser.add_argument("--env_name", default='Gridworld_SAS', help="Environment to run the code")
        parser.add_argument("--n_actions", default=16, help="number of base actions for gridworld", type=int)
        parser.add_argument("--difficulty", default=1, help="Difficulty for six room task", type=int)

        parser.add_argument("--max_episodes", default=int(1e4), help="maximum number of episodes (75000)", type=int)
        parser.add_argument("--max_steps", default=150, help="maximum steps per episode (500)", type=int)


    def SAS_PG(self, parser):
        parser.add_argument("--q_lr", default=1e-3, help="Learning rate for Q", type=float)
        parser.add_argument("--v_lr", default=1e-2, help="Learning rate for V", type=float)
        parser.add_argument("--alpha_rate", default=0.9999, help="Mixing momentum", type=float)
        parser.add_argument("--action_prob", default=0.8, help="Action available probability", type=float)
        parser.add_argument("--SF_bridge_prob", default=-1, help="Bridge available probability in SF map", type=float)
        parser.add_argument("--SAS_q_updates", default=16, help="Number of batches per optim step", type=int)

    def Main_AC_args(self, parser):
        parser.add_argument("--exp", default=0.075, help="Eps-greedy epxloration decay", type=float)
        parser.add_argument("--gamma", default=0.99, help="Discounting factor", type=float)
        parser.add_argument("--trace_lambda", default=0.9, help="Lambda returns", type=float)
        parser.add_argument("--actor_lr", default=1e-4, help="Learning rate of actor", type=float)
        # parser.add_argument("--critic_lr", default=1e-2, help="Learning rate of critic/baseline", type=float)
        parser.add_argument("--state_lr", default=1e-3, help="Learning rate of state features", type=float)

        parser.add_argument("--fourier_coupled", default=True, help="Coupled or uncoupled fourier basis",
                            type=self.str2bool)
        parser.add_argument("--fourier_order", default=3, help="Order of fourier basis, " +
                                                               "(if > 0, it overrides neural nets)", type=int)
        parser.add_argument("--NN_basis_dim", default='256', help="Shared Dimensions for Neural network layers")
        parser.add_argument("--Policy_basis_dim", default='2,16', help="Dimensions for Neural network layers for policy")

        parser.add_argument("--buffer_size", default=int(1e4), help="Size of memory buffer (3e5)", type=int)
        parser.add_argument("--batch_size", default=16, help="Batch size", type=int)
