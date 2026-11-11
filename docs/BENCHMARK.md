# Snow Lab — comparaison du moteur de neige

Page statique de la branche `main`, publiable par GitHub Pages avec **Settings → Pages → Deploy from a branch → main /docs**.

URL après activation : `https://leye0.github.io/PooLabVR/`.

## Ce qui est réellement comparé

Le navigateur exécute une simulation de particules WebGL2 (transform feedback) et affiche les FPS effectifs en direct. La résolution initiale est **200 × 200 pixels par œil**, avec choix de **400 × 400** et **800 × 800** ; changer la résolution n'enlève aucun flocon de la simulation. Le sélecteur de population permet 50 000, 200 000, 500 000, 1 million ou 2 millions de particules. Le bouton de mesure repart du **même état initial**, préchauffe chaque mode pendant 0,5 seconde, puis mesure 2 secondes par mode, à la résolution actuellement sélectionnée.

| Mode | Vent | Dessin stéréo |
| --- | --- | --- |
| Original | `sin`/`cos` par particule | Deux appels de dessin, un par œil |
| Stéréo groupée | `sin`/`cos` par particule | Un appel instancié pour deux yeux |
| Vent localisé | 64 × 64 cellules et 32 classes de phase | Deux appels de dessin |
| Combiné | Vent localisé | Un appel instancié |

Le champ de vent partagé regroupe **131 072 échantillons par image**, plutôt que deux évaluations trigonométriques indépendantes par particule. Chaque particule possède toujours sa position, sa vitesse, sa phase et son facteur de vent ; **les trajectoires ne sont plus identiques** à celles du modèle original. Cette optimisation est donc approximative, pas « lossless » au sens de trajectoires identiques.

La stéréo WebGL2 regroupe des appels de dessin avec `drawArraysInstanced`, mais exécute encore une projection par œil. La version expérimentale native DX12/HLSL (`SnowSplatStereo`) lit un état de particule une fois et traite les projections des deux yeux dans le même thread. Les deux approches ne sont pas directement équivalentes.

**Limites :** ce benchmark web n'exécute pas le moteur original, CUDA, DirectX 12, PhysX, les shaders DX12 du projet, ni le casque VR. Les FPS sont plafonnés par la synchronisation du navigateur et peuvent varier avec l'appareil. La colonne de temps GPU n'est disponible que lorsque le navigateur fournit l'extension `EXT_disjoint_timer_query_webgl2`. Une amélioration de FPS sur le site **ne prouve pas** une amélioration de la latence VR globale ni un objectif de 72 Hz atteint.

## Expérience native (opt-in)

Les chemins par défaut restent inchangés. Sur Windows avec le SDK et les dépendances du projet :

- `LABOVR_EXPERIMENT_CELL_WIND=1` active le champ CUDA mutualisé de 64 × 64 × 32 échantillons, recalculé sur le flux de neige.
- `LABOVR_EXPERIMENT_STEREO_SNOW=1` active le shader DX12 `SnowSplatStereo` et une seule dimension de dispatch en Y au lieu d'une par œil.
- Les deux variables peuvent être activées ensemble.

Ces changements natifs doivent encore être **compilés, exécutés et profilés sur une machine Windows compatible CUDA/DX12** ; un gain GPU n'est pas garanti (accès mémoire, occupation GPU, dépendances et files d'attente peuvent dominer).

## Comparaison CPU reproductible (sans CUDA)

`tools/SnowCpuCompare.cpp` est une référence numérique isolée. Elle fait évoluer deux millions d'états indépendants durant 48 images sur deux secondes, puis projette la neige sur 200 × 200 pixels. Elle vérifie que le regroupement stéréo conserve les mêmes pixels et états, et mesure l'écart introduit par le vent mutualisé.

```bash
g++ -O3 -std=c++20 -Wall -Wextra -Wpedantic tools/SnowCpuCompare.cpp -o /tmp/SnowCpuCompare
/tmp/SnowCpuCompare --particles 2000000 --width 200 --height 200 --frames 48 --out /tmp/snow-comparison
```

Cette expérience CPU n'est **pas** une prédiction de performance GPU ; les boucles et l'organisation mémoire y sont différentes.

### Exécution de référence : 2 000 000 particules, 48 images, 200 × 200

Sur l'environnement CPU de développement (mesure ponctuelle, sans répétitions ni intervalles de confiance) :

| Mode | Calcul physique | Projection | Temps total | Accélération vs original |
| --- | ---: | ---: | ---: | ---: |
| Original | 2 862 ms | 2 791 ms | 5 653 ms | 1,00× |
| Stéréo groupée | 2 785 ms | 2 000 ms | 4 785 ms | 1,18× |
| Vent localisé | 2 098 ms | 2 775 ms | 4 872 ms | 1,16× |
| Combiné | 2 192 ms | 2 088 ms | 4 280 ms | **1,32×** |

La stéréo groupée conserve exactement les états et pixels de cette référence CPU. Le vent localisé a produit un écart RMS final de **0,021827 m** sur la position des particules (en tenant compte de la périodicité spatiale) et un écart absolu moyen de **1,231776 / 255** sur les composantes d'image. L'apparence générale reste proche, mais ce n'est **pas** une simulation mathématiquement identique.
