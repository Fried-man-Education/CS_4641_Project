# Project Proposal
Team 10's topic is an exploration into the use of machine learning to predict future mutations of the COVID-19 genome. As of June 14th, 2022, over 500 million people have contracted COVID-19. Of these people, over 6.2 million of them have died. With the death of loved ones, social isolation, and economic downturn, it is not a stretch to say that the vast majority of the world has been negatively impacted by the events of this pandemic. This exploration aims to provide a tool to researchers, warning to governments, and guide travelers about emerging variants all in order to save lives and be one step closer in ending the pandemic.


While there are few publications on Covid-19 mutation prediction using Machine Learning, the general field of viral mutation prediction using Machine Learning is well-researched by academia. In our method, we will incorporate the methodology used in general viral mutation prediction with the characteristics of COVID-19(SARS-CoV-2) genome. In this project, we define a mutation as the changes in the amino acid sequence in the Spike Protein of individual families of COVID-19 variant (de Hoffer et al., 2022). 

The dataset that will be used in this project is the National Center for Biotechnology Information(NCBI) . We will be NCBI’s genetic sequence of all families of Covid-19 variants and mutations and divide the dataset into two parts, with the first 80% for training and the last 20% for testing. 


One potential method is using artificial neural network to identify changes within the genetic code in both DNA and RNA of the corresponding amino acid sequence. Every feature in the input is a nucleotide in the genetic sequence corresponding to a feature in the output. The training of the machine learning technique is fed with numerically encoded genetic sequences([A,C,G,T] -> [0,1,2,3]) of successive generations of the same Covid-19 family ​​(Salama et al., 2016). 

Another potential method is using recurrent neural network with Long-Short Term Memory. The feature set would be similar to the method above, with the changes of splitting the sequence into overlapping k-meters of length 4 by the use of a sliding window. The layering of our model will follow the methodology used in Deif(2021) and Yin(2020). 

Both methods would be performed with Scikit-learn and PyTorch.

Our models would be evaluated based on its accuracy in comparison with the actual mutation history of the Spike Protein of Covid-19. The accuracy is measured by determining the percentage of exact matching in the predicted sequence with the actual sequence.
## Links
- [Timeline](https://docs.google.com/spreadsheets/d/1nTeB63nvPim6VD8VA3zFnEYTLGWwTaIt4XnW-lwcBYs/edit?usp=drivesdk)
- Video Presentation
