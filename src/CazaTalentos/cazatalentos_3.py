import numpy as np
import time

np.random.seed(23452345)

MAXIMUM_AMOUNT_OF_SHOTS = 14000-1

ITERATIONS = 10000

WINNER_PROBABILITY = 0.7

PRINT_INFO = False

MAGIC_FACTOR = ((6/4) + (7/3))/2

SIMULATIONS = np.random.rand(ITERATIONS, MAXIMUM_AMOUNT_OF_SHOTS)

HITTING_POINTS = 1.0
MISSING_POINTS = - MAGIC_FACTOR * 1.0


class Player:

    def __init__(self, probability: float):
        assert 0 <= probability <= 1
        self.probability = probability
        self.score = 0.0


def clean_players(players: [Player]):
    for player in players:
        player.score = 0.0


def simulate_and_update_player(player: Player, random_uniform_shot: float):
    player_point: bool = random_uniform_shot < player.probability
    if player_point:
        player.score += HITTING_POINTS
    else:
        player.score += MISSING_POINTS


def get_players() -> [Player]:
    generated_players: [Player] = []
    for _probability in [x / 1000 for x in range(501, 600)]:
        generated_players.append(Player(probability=_probability))
    generated_players.append(Player(WINNER_PROBABILITY))
    np.random.shuffle(generated_players)
    return generated_players


def simulate_strategy_and_get_the_winner(players: [Player], simulations) -> Player:

    scores = set()
    score_to_players = {}  # I had to do all this because Python is REALLY slow :/
    # This is a dict of scores as keys and list of players as values
    for player in players:
        scores.add(player.score)
        if player.score not in score_to_players:
            score_to_players[player.score] = [player]
        else:
            score_to_players[player.score].append(player)

    for _iteration in range(int(MAXIMUM_AMOUNT_OF_SHOTS)):
        # we make the 'best one' shoot one time
        best_score: float = max(scores)
        best_player: [Player] = score_to_players[best_score][-1]
        simulate_and_update_player(best_player, simulations[_iteration])
        scores.add(best_player.score)
        if best_player.score not in score_to_players:
            score_to_players[best_player.score] = [best_player]
        else:
            score_to_players[best_player.score].append(best_player)
        score_to_players[best_score].pop()
        if len(score_to_players[best_score]) == 0:
            score_to_players.pop(best_score)
            scores.remove(best_score)

    # Now the prediction is the one who scored the most
    return score_to_players[max(scores)][-1]


start_time = time.time()

amount_of_good_predictions = 0
PLAYERS: [Player] = get_players()
for iteration in range(ITERATIONS):
    winner: Player = simulate_strategy_and_get_the_winner(players=PLAYERS, simulations=SIMULATIONS[iteration])
    amount_of_good_predictions += winner.probability == WINNER_PROBABILITY
    clean_players(PLAYERS)

print(f'Out of {ITERATIONS} iterations, there was {100*amount_of_good_predictions/ITERATIONS} percentage of success')

print(f'Total time used (in seconds): {time.time()-start_time}.')
