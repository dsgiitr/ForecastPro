function generateChart() {
  fetch('try.json')
    .then(response => response.json())
    .then(data => {
      let prices = data.prices;
      let trainPredict = data.trainPredict;
      let validPredict = data.validPredict;

      let graph_plot = document.getElementById("validation_graph");
      Plotly.newPlot(
        graph_plot,
        [
          { y: prices, name: "Actual Price" },
          { y: trainPredict, name: "Training" },
          { y: validPredict, name: "Validation" }
        ]
      );
    })
    .catch(error => {
      console.error('Error loading JSON file:', error);
    });
}
