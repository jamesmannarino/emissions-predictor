// const axios = require('axios');
// console.log("Hello TensorFlow");

async function getData() {
  const bldgData = await fetch(
    "https://data.cityofnewyork.us/resource/28fi-3us3.json?largest_property_use_type=Multifamily Housing&$where=starts_with(bbl_10_digits, '1')&$select=dof_gross_floor_area_ft,total_ghg_emissions_metric,address_1_self_reported&$limit=20000"
  );
  const bldgDataJSON = await bldgData.json();
  //turning strings into numbers
  const cleanedData = bldgDataJSON
    .map((bldg) => ({
      emissions: Number(bldg.total_ghg_emissions_metric),
      squareFootage: Number(bldg.dof_gross_floor_area_ft),
    }))
    //filtering outlier cases, i.e. buildings over 800,000 sqft and more than 6,000 tons of ghg emissions/year
    .filter((bldg) => bldg.emissions != 0
    && bldg.emissions < 6000
    && bldg.emissions != undefined && bldg.squareFootage != 0
    && bldg.squareFootage < 800000
    && bldg.squareFootage != undefined);

    //assigning values to x and y axes
    const values = await cleanedData.map((d) => ({
      x: d.squareFootage,
      y: d.emissions,
    }));

  return cleanedData;
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

  model.add(tf.layers.dense({units: 3}));
  model.add(tf.layers.dense({units: 3}));
  // model.add(tf.layers.dense({units: 15, activation: 'sigmoid'}));

  // // Add an output layer
  model.add(tf.layers.dense({ units: 1 }));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * Emissions on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any intermediate tensors.

  return tf.tidy(() => {

    tf.util.shuffle(data);


    const inputs = data.map((d) => d.squareFootage);
    const labels = data.map((d) => d.emissions);
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]).clipByValue(0, 900000);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]).clipByValue(0, 5000);


    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));

    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  });

  const batchSize = 32;
  const epochs = 10;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.squareFootage,
    y: d.emissions,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Square Footage",
      yLabel: "Emissions",
      height: 300,
    }
  );
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map((d) => ({
    x: d.squareFootage,
    y: d.emissions,
  }));

  tfvis.render.scatterplot(
    { name: "Square Footage v Emissions" },
    { values },
    {
      xLabel: "Square Footage",
      yLabel: "Emissions",
      height: 300,
      zoomToFit: false
    }
  );

  // More code will be added below
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);
  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log("Done Training");

  // Make some predictions using the model and compare them to the
// original data
  testModel(model, data, tensorData);
}

document.addEventListener("DOMContentLoaded", run);

  // const bldgData = await fetch(
  //   "https://data.cityofnewyork.us/resource/28fi-3us3.json?$where=starts_with(bbl_10_digits, '1')&$select=dof_gross_floor_area_ft,total_ghg_emissions_metric&$limit=20000"
  // );
  // const bldgData = await fetch(
  //   "https://data.cityofnewyork.us/resource/28fi-3us3.json?largest_property_use_type=Office&$where=starts_with(bbl_10_digits, '1')&$select=dof_gross_floor_area_ft,total_ghg_emissions_metric,address_1_self_reported&$limit=20000"
  // );

    //   const bldgData = await fetch(
  //   "https://data.cityofnewyork.us/resource/28fi-3us3.json?$where=starts_with(bbl_10_digits, '1')&$select=dof_gross_floor_area_ft,total_ghg_emissions_metric&$limit=20000"
  // );
