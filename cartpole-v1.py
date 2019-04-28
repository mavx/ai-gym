import json
import os

import gym
import numpy as np


class CartPole:
    def __init__(self):
        self._env = gym.make("CartPole-v1")
        self.best_strategy = None
        self.high_score = 0
        self.decision_spectrum = {
            'min': 0.0,
            'max': 0.0
        }
        self.decision_threshold = 0
        self.config_file = 'cartpole-v1' + '.json'
        self.loaded_config = {}

    def run_episode(self, train=True):
        env = self._env
        observation = env.reset()
        score = 0
        strategy = np.random.random(4) if train else self.best_strategy

        for i in range(1, 1000):
            env.render()

            decision = strategy @ np.array(observation)
            action = 0 if decision < self.decision_threshold else 1

            if train:
                self.update_decisioning(decision)

            observation, reward, done, info = env.step(action)
            score += reward

            if i % 100 == 0:
                print("Reached {} steps!".format(i))

            if done:
                observation = env.reset()
                break

        # Update states
        if train and (score > self.high_score or score == 500):
            self.high_score = score
            self.best_strategy = strategy
            print("New high score ({}) achieved with strategy - {}".format(score, strategy))

        print("\n#### EPISODE SUMMARY ####")
        print("CurrentScore: {}".format(score))
        print("HighScore: {}".format(self.high_score))
        print("DecSpec: {}".format(self.decision_spectrum))
        print("Threshold: {}".format(self.decision_threshold))

    def update_decisioning(self, decision: float):
        # Update stats
        if decision < self.decision_spectrum['min']:
            print("Updating MIN: {} -> {}".format(self.decision_spectrum['min'], decision))
            self.decision_spectrum['min'] = decision
        elif decision > self.decision_spectrum['max']:
            print("Updating MAX: {} -> {}".format(self.decision_spectrum['max'], decision))
            self.decision_spectrum['max'] = decision

        # Update threshold
        new_threshold = round((self.decision_spectrum['min'] + self.decision_spectrum['max']) / 2, 4)
        if new_threshold != self.decision_threshold:
            self.decision_threshold = new_threshold
            print("Updated decision threshold: {}".format(new_threshold))

    def run_for(self, num_episodes=10):
        for i in range(num_episodes):
            print("\n\nTRIAL # {}".format(i))
            self.run_episode()
        self.end()
        self.generate_report()

    def generate_report(self):
        print("\n\n#### OVERALL REPORT ####")
        print("HighScore: {}".format(self.high_score))
        print("Strategy: {}".format(self.best_strategy))
        print("DecisionSpec: {}".format(self.decision_spectrum))
        print("Threshold: {}".format(self.decision_threshold))

    def end(self):
        self._env.close()

    def save_config(self):
        config = {
            'decision_threshold': self.decision_threshold,
            'decision_spectrum': self.decision_spectrum,
            'high_score': self.high_score,
            'best_strategy': self.best_strategy.tolist()
        }
        last_high_score = self.loaded_config.get('high_score', 0)

        to_save = True
        if self.high_score <= last_high_score:
            to_save = input(
                "Best strategy only achieved {}/{}, save? (y/n)".format(self.high_score, last_high_score)) == 'y'

        if to_save:
            with open(self.config_file, 'w') as o:
                o.write(json.dumps(config, indent=2, default=str))
            print("Saved config to {}".format(self.config_file))

    def load_config(self):
        print(self.config_file)
        try:
            if os.path.isfile(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.loads(f.read())
                self.decision_threshold = config.get('decision_threshold', 0)
                self.decision_spectrum['min'] = config.get('decision_spectrum', {}).get('min', 0.0)
                self.decision_spectrum['max'] = config.get('decision_spectrum', {}).get('max', 0.0)
                self.best_strategy = config.get('best_strategy')
                self.loaded_config = config
                print("Loaded config from {}".format(self.config_file))
            else:
                print("Cannot find config from {}".format(self.config_file))
        except Exception as e:
            print(str(e))


def testrun():
    c = CartPole()
    c.load_config()
    c.run_episode(train=False)
    c.end()


def main():
    c = CartPole()
    c.load_config()
    c.run_for(num_episodes=500)
    c.save_config()


if __name__ == '__main__':
    main()
    # testrun()
