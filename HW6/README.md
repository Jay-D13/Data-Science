<!--- 
# Université de Montréal
# IFT-6758-A  -  A23  -  Science des Données
-->

# Devoir 6

Évaluation du devoir :

| Composant                                                   | Fichiers Requis  | Score |
|-------------------------------------------------------------|------------------|:-----:|
| Code (5 fonctions)                                          | `nlp_code.py`    |  50   |
| &emsp;+ figures, exécutions de cellules, sorties            | `hw6.ipynb`      |  10   |
| rapport (T1.3, T3 1-3, T4.4, T5)                            | `hw6.pdf`        |  40   |



Une partie de votre devoir sera évaluée automatiquement, c'est-à-dire que vous ne devez **pas modifier la signature des fonctions définies** (mêmes entrées et sorties).

### Soumission

Pour soumettre les fichiers, veuillez soumettre **uniquement les fichiers requis** (indiqués dans le tableau ci-dessus) que vous avez complétés à **gradescope**; ne pas inclure de données ou d'autres fichiers divers.

**Avertissement 1 : Vérifiez attentivement la manière dont j'ai défini les fonctions et leurs sorties attendues.**

**Avertissement 2 : Je vous demande d'effectuer certaines actions dans le notebook Jupyter, ne les sautez pas. Si je suis suspicieux quant à votre travail, je vérifierai que vous avez suivi mes instructions.**

**Avertissement 2 : Pour la tâche 5, vous devrez implémenter un modèle transformateur. Je vous recommande de réaliser cette partie dans Google Colab et d'utiliser un environnement d'exécution qui utilise des GPUs (vous pouvez déboguer votre implémentation sur CPU puis effectuer l'entraînement sur l'environnement d'exécution GPU).**

**Avertissement 3 : distilBERT a besoin d'un format spécifique pour ses entrées. Vous devrez changer le nom des colonnes de l'ensemble de données en "text" et "labels" sinon cela ne fonctionnera pas. De plus, je veux que vous conserviez les divisions train et test fournies dans les sections précédentes et que vous travailliez avec les critiques prétraitées et les étiquettes encodées (il est judicieux de les sauvegarder sous forme de fichiers CSV que vous pourrez utiliser ultérieurement).**

**Avertissement 4 : Soyez attentif au mot "discussion" ou à la balise [Discussion]. Je n'accepterai pas les réponses par oui/non ou les réponses avec un effort minimal pour le rapport. Appuyez votre analyse avec les graphiques demandés et utilisez des références lorsque nécessaire.**

**Avertissement 5 : Dans votre rapport, incluez votre nom et votre numéro d'étudiant. Si une tâche comporte des questions auxquelles vous devez répondre ou quelque chose que vous devez discuter ou expliquer, veuillez ajouter un titre indiquant la section rapportée de l'assignation, puis vos réponses/discussions.**

## Tâche 1 : Préparation des données et Exploration initiale

Nous commencerons par charger nos données et vérifier leur forme générale et leur contenu. Vous devrez vérifier les valeurs NaN et les lignes en double et les gérer correctement (c'est-à-dire en conservant la première occurrence et en supprimant les autres). Assurez-vous de vérifier que nous travaillons avec des points de données uniques.

- Complétez les opérations requises pour répondre aux questions du rapport `hw6.ipynb`
- Répondez dans votre rapport aux questions suivantes `hw6.pdf` :

* Nombre de points de données (?)
* Combien de valeurs uniques notre colonne cible contient-elle ?
* Contient-il des valeurs NaN ?
* Y a-t-il des critiques en double ? (Si vous trouvez des lignes en double, combien de doublons avez-vous trouvés ? 

## Tâche 2 : Prétraitement des données

- Complétez `nlp_code.py:preprocess_text()`
- Complétez les opérations requises dans le notebook `hw6.ipynb`

Dans cette section, nous prétraiterons nos critiques. Vous devrez compléter la fonction `preprocess_text()` qui prend une seule chaîne de caractères et la formate. Vous appliquerez cette fonction à toutes les critiques.

## Tâche 3 : Analyse exploratoire des données (AED)

Dans cette section, nous réaliserons une analyse exploratoire et répondrons à certaines questions sur nos critiques et notre ensemble de données. N'oubliez pas d'effectuer les opérations nécessaires et de soutenir votre réponse avec les supports nécessaires.

- Complétez `nlp_code.py:review_lengths()`
- Complétez `nlp_code.py:word_frequency()`
- Complétez les opérations requises dans le notebook `hw6.ipynb`
- Répondez dans votre rapport aux questions suivantes et appuyez-les avec les graphiques et l'analyse requis (`hw6.pdf`) :

* Comment sont distribuées les valeurs cibles ? Avons-nous un ensemble de données presque équilibré ?
* Toutes les critiques ont-elles la même longueur ?
* Quelle est la longueur de séquence moyenne ?
* Quels sont les 20 mots les plus fréquents ?
* Quels sont les 20 mots les moins fréquents ?
* Après avoir effectué une AED. Pensez-vous qu'il sera facile de classifier ces critiques ? Pourquoi oui ? / Pourquoi pas ? **[Discussion]**


## Tâche 4 : Extraction de caractéristiques et Préparation de la cible

Dans cette section, nous encoderons notre colonne cible (vous devrez implémenter la fonction `encode_sentiment()`, soyez attentif au type de données de "sentiment" encodé) et effectuerons une extraction de caractéristiques avec un vectoriseur Tf-idf. Nous entraînerons un modèle simple et l'utiliserons comme référence pour comprendre la difficulté de classer les critiques. Enfin, vous apprendrez un nouvel outil d'explicabilité de modèle appelé LIME (vous devrez compléter la fonction `explain_instance()`).

- Complétez `nlp_code.py: encode_sentiment()`
- Complétez `nlp_code.py:explain_instance()`
- Complétez les opérations requises dans le notebook `hw6.ipynb`
- Répondez dans votre rapport aux questions suivantes et appuyez-les avec les graphiques et l'analyse requis `hw6.pdf` :

Tf-idf (Term-frequency times inverse document-frequency):

* Expliquez les désavantages d'utiliser cette méthode. **[Discussion]**

* Fournissez une méthode alternative que nous aurions pu utiliser pour trouver une meilleure représentation numérique des mots présents dans notre corpus. (Pourquoi pensez-vous que cela pourrait fonctionner mieux ?) **[Discussion]**

LIME (Local Interpretable Model-agnostic Explanations):

Dans votre rapport, ajoutez la visualisation obtenue et fournissez une interprétation pour celle-ci. **[Discussion]**


## Tâche 5 : Exploration d'un modèle Transformer

- Implémentez le modèle distilBERT et obtenez les scores et la visualisation requis : `hw6.ipynb`
- Laissez des preuves de votre implémentation dans le notebook Jupyter : `hw6.ipynb`
- Incluez dans votre rapport l'analyse mentionnée aux points 1, 2 et 3 : `hw6.pdf`

Dans cette section, je voudrais que vous `essayiez de` battre notre modèle précédent avec un modèle beaucoup plus complexe. Vous utiliserez un modèle transformateur appelé `distilBERT` pour cela (vous pouvez utiliser la bibliothèque HuggingFace pour l'implémenter). Dans votre rapport, je voudrais que vous :

1-. Implémentiez le modèle `distilBERT`, l'entraîniez et l'évaluiez sur le même ensemble d'entraînement que notre `classifieur Naïve Bayes`. Vous pouvez sauvegarder les ensembles de données prétraités sous forme de fichiers `csv` (avec les noms de colonnes `text`et `labels`) puis les charger à l'aide de la méthode `load_dataset` de la bibliothèque `datasets`. Vous devez faire attention au formatage des entrées pour ce modèle transformateur (son nom est : `distilbert-base-uncased`). 

2-. Explorez 2 différentes façons de régler le modèle (# epochs, learning rate, weight decay, etc.) pour améliorer les performances de classification (incluez le(s) tableau(x) des résultats obtenus par rapport aux approches explorées, indiquez la configuration qui a produit les meilleurs résultats). Je veux connaître en détail la méthodologie que vous avez suivie pour améliorer les performances du modèle. Par conséquent, j'attends une discussion raisonnable sur les approches que vous avez prises (je retirerai des points pour les changements aléatoires des hyperparamètres du modèle). **[Discussion]**

3-. Vous devrez inclure dans votre rapport votre précision, rappel, et scores F1 (discutez-les et comparez-les). De plus, vous devez inclure l'image de votre matrice de confusion sous forme de heatmap (meilleur modèle). Rapportez si les résultats obtenus sont bien meilleurs que ceux obtenus avec le `classifieur Naïve Bayes`. (Soutenez vos commentaires en comparant les scores et les heatmaps, j'attends une bonne quantité de discussion). **[Discussion]**
