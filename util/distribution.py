from random import uniform


class Distribution(dict):
    """ A distribution of items and their associated probabilities. """

    def __init__(self, args=None):
        """
        Initializes state distribution from list or given distributions.
        """
        if type(args) is list:
            dict.__init__(self, {item: prob for item, prob in args})
        elif args:
            super(Distribution, self).__init__(args)
        else:
            super(Distribution, self).__init__()

    def expectation(self, values, require_exact_keys=True):
        """
        Returns an expectation over values using the probabilities in this distribution.
        """
        if require_exact_keys:
            assert self.keys() == values.keys(), \
                'Conditional probabilities keys do not map to distribution.\n' + \
                str(set(values.keys())) + ' != ' + str(self.keys())
            return sum(values[key] * self[key] for key in self)
        else:
            return sum(values[key] * self[key] for key in (self.keys() & values.keys()))

    def conditional_update(self, conditional_probs):
        """
        Given a set of conditional probabilities, the distribution updates itself via Bayes' rule.
        """
        assert self.keys() == conditional_probs.keys(), \
            'Conditional probabilities keys do not map to distribution.\n' + \
            str(set(conditional_probs.keys())) + ' != ' + str(self.keys())

        new_dist = self.copy()

        for key in conditional_probs:
            new_dist[key] *= conditional_probs[key]

        return new_dist.normalize()

    def __add__(self, other_distribution):
        """
        Sums two distributions across keys.
        """
        #assert self.keys() == other_distribution.keys(), \
        #    'Conditional probabilities keys do not map to distribution.\n' + \
        #    str(set(other_distribution.keys())) + ' != ' + str(self.keys())

        new_dist = Distribution()

        for key in self.keys() | other_distribution.keys():
            new_dist[key] = self.get(key, 0.0) + other_distribution.get(key, 0.0)

        return new_dist

    def __radd__(self, other):
        """ Reverse add for cases when using 'sum' over Distributions. """
        if other == 0:  # When using 'sum', the first operation tries 0 + obj, then obj + 0
            return self
        else:
            return self.__add__(other)

    def __mul__(self, num):
        """ Scale all values up by a given factor. """
        assert type(num) == float or type(num) == int, 'Incorrect use of product between Distribution and ' + str(type(num))

        new_dist = Distribution()
        for key, value in self.items():
            new_dist[key] = value * num

        return new_dist

    def __rmul__(self, num):
        """ Scale all values up by a given factor. """
        return self.__mul__(num)

    def normalize(self):
        """
        Normalizes the distribution such that all probabilities sum to 1.
        """
        total = sum(self.values())

        assert total > 0, 'Distribution probability total = 0.'

        for item in self:
            self[item] /= total

        return self

    def sample(self):
        """
        Returns a probabilistically selected item from the distribution.
        """
        target = uniform(0, sum(self.values()))  # Corrected to sum of probabilities for non-normalized distributions.
        cumulative = 0

        # Accumulate probability until target is reached, returning state.
        for item, probability in self.items():
            cumulative += probability
            if cumulative > target:
                return item

        # Small rounding errors may cause probability to not reach target for last state.
        return item

    def __repr__(self):
        return '\nDistribution {\n' + '\n'.join(str(key) + '\n\tP=' + str(val) + '\n' for key, val in self.items()) + '}'

    def __missing__(self, key):
        self[key] = 0.0
        return 0.0

    @staticmethod
    def uniform(items):
        probability_per_item = 1.0/len(items)
        return Distribution({item: probability_per_item for item in items})
