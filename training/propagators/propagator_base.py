


class PropagatorBase(object):
    def __init__(self, loss_func, optimiser, networks, normaliser):
        self.networks = networks
        self.normaliser = normaliser
        self.optimiser = optimiser
        self.loss_func = loss_func
        self.epoch = 0.

    def eval(self):
        for network_name, network in self.networks.items():
            network.eval()

    def train(self):
        for network_name, network in self.networks.items():
            network.train()

    def set_epoch(self, e):
        self.epoch = e

    def train_forward_backward(self, samples):
        raise NotImplementedError()

    def testing_forward(self, samples):
       raise NotImplementedError()

    def get_state_dicts(self):
        state_dicts ={}

        for network_name, network in self.networks.items():
            state_dicts[network_name] = network.state_dict()

        state_dicts['optimizer_state_dict'] = self.optimiser.state_dict()

        return state_dicts
