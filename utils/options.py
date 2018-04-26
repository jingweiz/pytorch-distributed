class Params(object):
    def __init__(self):
        self.verbose     = 0            # 0(warning) | 1(info) | 2(debug)

        # training signature
        self.machine     = "aisdaim"    # "machine_id"
        self.timestamp   = "18042500"   # "yymmdd##"
        # training configuration
        self.mode        = 1            # 1(train) | 2(test model_file)
        self.config      = 5

        self.seed        = 123
        self.render      = False        # whether render the window from the original envs or not
        self.visualize   = True         # whether do online plotting and stuff or not
        self.save_best   = False        # save model w/ highest reward if True, otherwise always save the latest model

        self.num_envs_per_actor = 4     # NOTE: must be 1 for envs that don't have parallel support
        self.num_actors = 2
        self.num_learners = 1


class ModelParams(Params):
    def __init__(self):
        super(ModelParams, self).__init__()


class AgentParams(Params):
    def __init__(self):
        super(AgentParams, self).__init__()



class Options(Params):
    model_params = ModelParams()
    agent_params = AgentParams()
