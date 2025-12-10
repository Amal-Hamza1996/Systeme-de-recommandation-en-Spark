# Project Objective

The goal of this project is to design a Big Data application and deploy it within a computing infrastructure. The project consists of two main parts: one focusing on the design of a data processing application, and the other on setting up the computing infrastructure.

# Big Data Application Design

The purpose of this part is to implement and evaluate a gradient descent method to solve a collaborative filtering problem. We have a file containing ratings for a number of movies by different users on a platform. Since these users have not rated, or even watched, all the available movies, the goal is to estimate the “missing” ratings from the available ratings, movies, and users.

The data is stored as a matrix \(R = [r_{ij}]\) (rows → users, columns → movies, matrix entries → ratings). This problem can be modeled as finding a low-rank factorization of \(R\), which leads to the following optimization problem: 

![Optimization](https://user-images.githubusercontent.com/38117821/137370917-eeaed90e-6db2-4073-a90e-028ba6810d9c.PNG)

The objective is to find the pair of matrices \(P\) and \(Q\), of fixed rank, such that the product of \(P\) and the transpose of \(Q\) minimizes the difference with the observed data \(R\).

The data file `ratings.dat` must be located in the directory `/hadoop-cluster-docker/data/`.

# Computing Infrastructure Setup

The goal of this part is to design a distributed Spark computing infrastructure, simulated on a personal computer using Docker containers. One container runs the Spark Master, and a set of containers run the Spark Slaves. The infrastructure is represented in the diagram below.

![Docker](https://user-images.githubusercontent.com/38117821/137381791-40c976be-bb7e-407f-afe4-007ca2f1f333.PNG)

The Slaves are registered with the Master. A client (a shell) can submit a Spark application for execution to the Master (using `spark-submit`), and the Master distributes the execution to the Slaves.
