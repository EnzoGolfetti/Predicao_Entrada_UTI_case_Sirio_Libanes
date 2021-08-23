# Ajudando na tomada de decisão para internações em leitos de UTI em decorrência da COVID-19

## Um projeto de Machine Learning para o Bootcamp de Ciência de Dados da Alura Cursos

Desde o começo da pandemia de COVID-19 os gestores e tomadores de decisão especialmente aqueles responsáveis por hospitais e portanto, diretamente por leitos de UTI vem tentando antever o máximo possível quando um paciente irá necessitar desse tipo de atendimento, de forma a um melhor acolhimento e garantir a sobrevivência destes.

A palavra antever lembra predição, predição nos leva a um novo normal do mundo: o uso generalizado de **algoritmos de aprendizado de máquina**. Eles estão por todos lugares, no celular ou computador no qual você está lendo esse README, nos smartwatches, na Netflix, Amazon Prime Video e no Instagram para citar alguns exemplos.

Pensando nessa generalização e no barateamento do acesso à essas tecnologias (esse projeto por exemplo foi feito num serviço gratuito do Google para Data Science chamado Google Colaboratory) a equipe de Data Science do Hospital Sírio Libanês decidiu publicar na tradicional plataforma social e de competições de *Machine Learning* Kaggle um dataset com informações sobre pacientes do hospital e lançaram um desafio para a comunidade: **"É possível prever quem necessitará de leitos de UTI e quem *não* necessitará a partir de algoritmos?".**

Visando uma aplicação real de *Machine Learning*, o 2º Bootcamp de Ciência de Dados da Alura propôs como projeto final esse mesmo desafio para os alunos. Este projeto sumariza meu aprendizado ao longo do curso e expõe o que ainda preciso melhorar. Outros notebooks específicos do curso podem ser conferidos nesse [link](https://github.com/EnzoGolfetti/bootcamp_ds_aplicada).

## O projeto
Acesse o notebook completo: [predicao_entrada_uti_sirio_libanes_v1.ipynb](https://github.com/EnzoGolfetti/Predicao_Entrada_UTI_case_Sirio_Libanes/blob/main/predicao_entrada_uti_sirio_libanes_v1.ipynb)

### Ferramental
Primeiramente para realização do desafio, precisamos escolher as ferramentas que seriam utilizadas, nisso seguindo o desenvolvimento do Bootcamp elegemos:
- Linguagem Python;
- Jupyter Notebook no ambiente do Google Colaboratory do Google;
- GitHub para armazenar os arquivos em nuvem;
- Bibliotecas python principais:
  -  Pandas -- para análise de dados;
  -  NumPy -- biblioteca para álgebra linear;
  -  Pyforest -- automaticamente importa mais de 40 bibliotecas famosas de Data Science em Python;
  -  pandas_profilling -- para gerar reports de dados de forma simples e versátil;
  -  Scikit Learn -- uma das principais bibliotecas de ML em Pyhon;
  -  Scikit-Optimizer - biblioteca para hiperparametrização com abordagem bayesiana;
  -  MLxtend -- para o método Stack and Embedded ML.
 
### Seções e métodos
  - Seção 1: Ferramentas em mãos, a primeira seção foi a de importação de todas as bibliotecas, leitura e armazenamento do dataset e definição das funções utilizadas;
  - Seção 2: Análise Exploratória dos Dados, nesta seção exploramos graficamente as informações disponibilizadas pelo Sírio-Libanês, observando correlações com internações na UTI, como gênero, idade e comorbidades.
  - Seção 3: Em seguida realizamos um processo bastante comum em projetos de Machine Learning, chamado *Feature Selection e Feature Engineering*, em que tentamos escolher as colunas com maior correlação com internação na UTI e amenizar possíveis ruídos no dataset que levassem a *bias* (viés) ou diminuíssem a capacidade dos algoritmos levando a *overfitting* ou *underfitting*. Aqui tivemos como *output* três datasets cada um com características únicas visando avaliar qual combinação de *features* levaria a melhor performance.
  - Seção 4: Construímos *benchmarks* de algoritmos de classificação como performance mínima para o algoritmo final, a ideia geral é a de que se não conseguimos "bater" o modelo mais simples possível então não se justifica todo o trabalho e gasto computacional com algoritmos altamente sofisticados.
  - Seção 5: Nesta pulamos para um outro notebook, onde rodamos alguns algoritmos iniciais visando ter um norte e um embasamento para escolha de algoritmos a serem testados e aprofundados. Com a biblioteca LazyPredict em mãos pudemos rodar mais de 30 algoritmos de uma vez avaliando qual deles performava melhor nos datasets construídos anteriormente.
  - Seção 6: De volta ao notebook principal, a partir dos 4 melhores algoritmos na avaliação com o Lazypredict nos aprofundamos neles: RandomForestClassifier, XGBClassifier, LGBMClassifier e BernoulliNB, avaliando algumas métricas principais como ROC-AUC Score, Precision, Accuracy, Recall, F1-Score (uma média harmônica entre Precision e Recall) e a matriz de confusão.
  - Seção 7: Continuando na escolha pelo algoritmo com a melhor performance, selecionamos os dois melhores: RandomForestClassifier e XGBClassifier e dedicamos esta seção inteira a uma forma de diminuir o problema da aleatoriedade no corte do dataset entre treino e teste, aplicamos uma função que rodava os modelos várias vezes (ou seja, com várias combinações) e computava o AUC médio e o Recall médio dos modelos;
  - Seção 8: Essa seção foi dedicada a encontrar os melhores hiperparâmetros para os dois modelos, a partir da abordagem bayesiana, após mais de 5 horas rodando o notebook conseguimos chegar a tal resultado como pode ser verificado no trabalho.
  - Seção 9: Num esforço final de chegar a um modelo ainda melhor, decidimos fazer a aplicação de um método conhecido como Stack and Embed Machine Learning, em que se treinam vários modelos em camadas e suas saídas se tornam inputs para o próximo e assim por diante até um meta-learner final que generaliza todas essas saídas e tenta fazer uma predição ainda mais eficaz.
  - Seção 10: Aqui salvamos o modelo e fazemos as devidas conclusões do projeto.     

### Conclusões
Após exaustivos testes e muito gasto computacional, concluímos que o Stacked Model foi o que performou melhor, como argumentamos ao longo do trabalho, apesar deste ter tido um ROC-AUC Score de 0.84 contra 0.86 do RandomForestClassifier tunado com os melhores hiperparâmetros, seu Recall foi expressivamente melhor com 0.82 contra 0.7 nos fazendo optar por ele, visto que o Recall mede a capacidade dos modelos de indicar corretamente a True Label como sendo True Label, ou seja, a capacidade de corretamente indicar que alguém necessita de um leito de UTI.

Nosso modelo final portanto, foi o **modelo Stack and Embedded com um ROC-AUC Score de 0.84 e um Recall de 0.82**

### Referências e agradecimentos
Todas as referências utilizadas estão disponíveis no notebook, nele mesmo discutimos mais profundamente alguns assuntos tratados aqui e nosso modelo treinado também está disponível no repositório deste projeto.

Agradeço a todos os instrutores do Bootcamp da Alura que sempre estiveram disponíveis para todas minhas dúvidas e que se empenharam para entregar um material que valeu a pena. Também a todos os alunos que ao compartilharem seus conhecimentos nas comunidade me ajudaram a solucionar dúvidas e problemas aos quais não tive a perspectiva e perspicácia no momento para resolver.

Acesse o notebook completo: [predicao_entrada_uti_sirio_libanes_v1.ipynb](https://github.com/EnzoGolfetti/Predicao_Entrada_UTI_case_Sirio_Libanes/blob/main/predicao_entrada_uti_sirio_libanes_v1.ipynb)

Enzo Golfetti - 23/08/2021
