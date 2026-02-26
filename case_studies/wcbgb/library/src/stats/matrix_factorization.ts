// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This module contains an implementation of the Community Notes matrix factorization
// algorithm, inteded for identifying points of common ground / consensus. This algorithm
// is based on fitting an approximation of the full vote / rating matrix according to a
// linear model as follows:
//
// $$ \hat{r}_{un} = \mu + i_u + i_n + f_u \cdot f_n $$
//
// Here, \hat{r}_{un} is the predicted rating for user u and note n, and f_u & f_n are
// user and note factor vectors, which can be thoughts of as preference embeddings.
// The terms i_u and i_n are the intercept terms for the given user and note, i_n in particular
// corresponding to how likely overall people are to agree with the given comment, irrespective
// of which side they fall on in relation to more polarizing comments.
// _These values_ are our "helpfulness" (or common ground / consensus) metrics.
//
// (The term /mu is an overall conversation intercept term, representing the average rating
// of a randomly selected rating, and while not of immediate focus, does capture a sense of
// the overall agreeableness of the conversation.)
//
// The parameters for this model are learned via gradient descent according to a regularized
// least squares error loss function:
//
// $$ \sum_{r_{un}} (r_{un} - \hat{r}_{un})^2 + \lambda_i (i_u^2 + i_n^2 + \mu^2) + \lambda_f (||f_u||^2 + ||f_n||^2) $$
//
// The terms \lambda_i and \lambda_f are the regularization terms on the intercepts and factors,
// respectively. To bias the model towards explaining variance with the user/note factors over
// the intercepts, \lambda_i is set 5 times higher than \lambda_f. This effectively raises the
// evidential bar for comments to recieve a high helpfulness score, improving robustness, especially
// with sparse data.

import * as tf from "@tensorflow/tfjs-core";

async function loadTFJS() {
  if (process.env.TFJS_NODE_GPU === "false") {
    await import("@tensorflow/tfjs");
    console.log("TFJS_NODE_GPU set to false, using CPU-only version");
  } else {
    try {
      await import("@tensorflow/tfjs-node-gpu");
      console.log("GPU version of tensorflow loaded");
    } catch (error) {
      console.warn("Failed to load GPU version of tensorflow:", error);
      console.warn("Falling back to CPU-only version");
      await import("@tensorflow/tfjs");
    }
  }
}

loadTFJS();

export interface Rating {
  userId: number;
  noteId: number;
  rating: number;
}

/**
 * Given ratings, return helpfulness scores and other model parameters for the given set of ratings.
 * @param ratings A collection of Rating values
 * @param numFactors The factor dimensionality
 * @param epochs Number of training iterations to run per learningRate
 * @param learningRate Either a single learning rate value, or an array of values for a learning rate schedule
 * @param lambdaI Intercept term regularization parameter
 * @param lambdaF Factor term regularization parameter
 * @returns Helpfulness scores
 */
export async function communityNotesMatrixFactorization(
  ratings: Rating[],
  numFactors: number = 1,
  epochs: number = 400,
  learningRate: number | number[] = [0.05, 0.01, 0.002, 0.0004],
  lambdaI: number = 0.15,
  lambdaF: number = 0.03
): Promise<number[]> {
  // infer numUsers as the max of all userId values in the ratings collection
  const numUsers =
    ratings.map((r) => r.userId).reduce((prev, current) => Math.max(prev, current), 0) + 1;
  const numNotes =
    ratings.map((r) => r.noteId).reduce((prev, current) => Math.max(prev, current), 0) + 1;

  // Initialize parameters randomly.  Using tf.variable allows us to update them during training.
  const mu = tf.variable(tf.scalar(0.0));
  const userIntercepts = tf.variable(tf.randomNormal([numUsers, 1]));
  const noteIntercepts = tf.variable(tf.randomNormal([numNotes, 1]));
  const userFactors = tf.variable(tf.randomNormal([numUsers, numFactors]));
  const noteFactors = tf.variable(tf.randomNormal([numNotes, numFactors]));

  const learningRates = typeof learningRate === "number" ? [learningRate] : learningRate;

  // Convert ratings to tensors for efficient computation.
  const userIds = tf.tensor1d(
    ratings.map((r) => r.userId),
    "int32"
  );
  const noteIds = tf.tensor1d(
    ratings.map((r) => r.noteId),
    "int32"
  );
  // For the math to work out, this should be a column vector, hence expand dims
  const ratingValues = tf.tensor1d(ratings.map((r) => r.rating)).expandDims(1);
  const lambdaIVar = tf.scalar(lambdaI);
  const lambdaFVar = tf.scalar(lambdaF);

  const loss = (): tf.Scalar => {
    // First collect all of the indices from the ratings matrix
    const ratingUInts = userIntercepts.gather(userIds);
    const ratingNInts = noteIntercepts.gather(noteIds);
    const ratingUFactors = userFactors.gather(userIds);
    const ratingNFactors = noteFactors.gather(noteIds);
    const muSquare = mu.square();

    // this is effectively the row-wise dotproduct; Note that expandDims is necessary
    // for tensor shapes to match up in later operations
    const factorDotProduct = ratingUFactors.mul(ratingNFactors).sum(1).expandDims(1);

    // compute the individual values \hat{r}_{un}
    const predictedRatings = mu.add(ratingUInts).add(ratingNInts).add(factorDotProduct);

    // compute the squared errors between predicted and actual ratings
    const squaredError = ratingValues.sub(predictedRatings).square();

    // the intercept regularization terms
    const interceptRegularization = lambdaIVar.mul(
      ratingUInts.square().add(ratingNInts.square()).add(muSquare)
    );

    // the factor regularization terms
    const factorRegularization = lambdaFVar
      .mul(ratingUFactors.euclideanNorm(1).square().add(ratingNFactors.euclideanNorm(1).square()))
      .expandDims(1);

    return squaredError.add(interceptRegularization).add(factorRegularization).sum();
  };

  // Training loop.
  for (const rate of learningRates) {
    console.log("Setting learning rate to:", rate);
    const optimizer = tf.train.adam(rate);
    for (let epoch = 0; epoch < epochs; epoch++) {
      optimizer.minimize(loss, true, [
        mu,
        userIntercepts,
        noteIntercepts,
        userFactors,
        noteFactors,
      ]);
      if ((epoch + 1) % 10 === 0) {
        console.log(`Epoch ${epoch + 1}, Loss: ${await loss().data()}`);
      }
    }
  }

  // Extract note intercepts (helpfulness scores).
  const helpfulnessScores = Array.from(await noteIntercepts.data());
  return helpfulnessScores;
}
