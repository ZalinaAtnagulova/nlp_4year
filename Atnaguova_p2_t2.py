from nltk.corpus import wordnet
from itertools import islice

#1) Найти все значения (синсеты) для лексемы glow
cup = wordnet.synsets('cup')
for ss in cup:
    print(ss, ss.definition())

#2) найти гиперонимы для значения лексемы cup - кубок
for ss in cup:
    print('Гипероним ', ss.hypernyms(), ' к значению ', ss.definition())

#3) вычислите одним из способов расстояние: (tea, coffee), (container, aretefact)
def get_dist_sim(ss1, lexeme):
    distances = []
    similarities = []
    for ss2 in lexeme:
        dist = ss1.shortest_path_distance(ss2)
        if dist is not None:
            distances.append(dist)
            sim = ss1.path_similarity(ss2)
            similarities.append(sim)
    return distances, similarities

tea = wordnet.synsets('tea')
coffee = wordnet.synsets('coffee')
container = wordnet.synsets('container')
artifact = wordnet.synsets('artifact')

# min d(tea, coffee)
dist1 = get_dist_sim(tea[0], coffee)[0]
print('min d(tea, coffee): {}'.format(min(dist1)))
print('closest lemma definition: {}\n'.format(coffee[dist1.index(min(dist1))].definition()))

# min d(container, artifact)
dist1 = get_dist_sim(container[0], artifact)[0]
print('min d(tea, coffee): {}'.format(min(dist1)))
print('closest lemma definition: {}\n'.format(artifact[dist1.index(min(dist1))].definition()))
