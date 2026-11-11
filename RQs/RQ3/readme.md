# RQ3
## Does stronger NDCG weighting change behavior? 
Use the four weights 0.5,1,1.5,2
0.5,1,1.5,20.5,1,1.5,2 as an ordered sweep. 	

Six-panel figure:
NDCG.
Recall.
Abstract cost.
Latency.
Immediate-STOP rate.
Mean number of enhancement operations.
All panels share the NDCG coefficient on the x-axis.


## Do terminal incentives change stopping? 

The most useful behavioral figure is first-action distribution:
x-axis: reward configuration.
y-axis: proportion of queries.
Segments: STOP, rerank, PRF, reformulate.

+ 

Title: Impact of Reward Variants on Retrieval Performance and Stopping Behavior

| Reward variant | Mean NDCG | Mean Recall | Mean cost | Mean latency | Immediate STOP (%) | Later explicit STOP (%) | Horizon reached (%) | Mean enhancement count |
|---|---|---|---|---|---|---|---|---|
| Base | — | — | — | — | — | — | — | — |
| Other variants | — | — | — | — | — | — | — | — |


## Cost and latency sweeps
A compact design is:
    Base coefficient.
    Zero coefficient.
    Twice the Base coefficient.
Run this separately for cost and latency, holding the other terms fixed. That is four additional configurations because Base is shared.
