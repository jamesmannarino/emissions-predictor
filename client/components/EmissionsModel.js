import React from 'react';
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

class EmissionsModel extends React.Component {
  constructor() {
    super()
    this.state = {
      squareFootage: null,
      emissions: null
    }
    this.getData = this.getData.bind(this)
  }
  async getData() {
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
      console.log(cleanedData)
      //assigning values to x and y axes
      const values = await cleanedData.map((d) => ({
        x: d.squareFootage,
        y: d.emissions,
      }));

    return cleanedData;
  }
  createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

    // model.add(tf.layers.dense({units: 3}));
    // model.add(tf.layers.dense({units: 3}));
    model.add(tf.layers.dense({units: 15, activation: 'sigmoid'}));

    // // Add an output layer
    model.add(tf.layers.dense({ units: 1 }));

    return model;
  }
  convertToTensor(data) {
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
  async trainModel(model, inputs, labels) {
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
  testModel(model, inputData, normalizationData) {
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
  render() {
    return (
    <div>
      {/* {async() => await this.getData()} */}
    </div>
    )
  }
}

export default EmissionsModel
