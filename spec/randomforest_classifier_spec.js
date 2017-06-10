/*
Copyright (c) 2012 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

var RandomForestClassifier = new require('../lib/apparatus/classifier/randomforest_classifier');
var fs = require('fs'),    readline = require('readline');
var iris = fs.readFileSync('./spec/data/iris.data').toString().split("\n");
var train_examples = [], test_examples = [], train_labels=[], test_labels=[];
for(var i=0;i<iris.length;i++) {
    var line = iris[i].split(",");
    let inputs = line.splice(0,4), label = line[0];

    for(var j=0;j<inputs.length;j++) {
        inputs[j] = parseFloat(inputs[j]);
    }

    if(i % 4 == 0) {
        test_examples.push(inputs);
        test_labels.push(label);
    } else {
        train_examples.push(inputs);
        train_labels.push(label);
    }
}

describe('randomforest', function() {

    it('should perform binary classification with 2d features', function() {           
        
        randomforest.addExample([-0.4326,  1.1909   ], 1);
        randomforest.addExample([1.5    , 3.0       ], 1);
        randomforest.addExample([0.1253 , -0.0376   ], 1);
        randomforest.addExample([0.2877 ,   0.3273  ], 1);
        randomforest.addExample([-1.1465,   0.1746  ], 1);
        randomforest.addExample([1.8133 ,   2.1139  ], -1);
        randomforest.addExample([2.7258 ,   3.0668  ], -1);
        randomforest.addExample([1.4117 ,   2.0593  ], -1);
        randomforest.addExample([4.1832 ,   1.9044  ], -1);
        randomforest.addExample([1.8636 ,   1.1677  ], -1);
        
        expect(randomforest.classify([-0.5 , -0.5 ])).toBe(1);
        
        // random forest are not deterministic, check on average it works
        var count = 0;
        for(var tests=0; tests<200; tests++){
            randomforest.train();
            if(randomforest.classify([1.0, 2.0]) == 1) {
                count++;
            }
        }
        expect(count).toBeGreaterThan(50); 
    }); 
    
    it('should correctly classify the IRIS dataset', function() {

        var randomforest = new RandomForestClassifier();
        
        for(var i = 0; i < train_examples.length; i++) {
          randomforest.addExample(train_examples[i], train_labels[i]);
        }
        
        randomforest.train();
        
        for(var i = 0; i < test_examples.length; i++) {
          expect(randomforest.classify(test_examples[i])).toBe(test_labels[i])
        }
    });
    
    it('should perform binary classification with 3d features', function() {

       var randomforest = new RandomForestClassifier();

        // generated from sklearn.datasets.make_classification
        var train_examples = [[ 0.31999894,  1.91372333, -1.98036028],
       [ 1.05453944, -1.21106664,  0.00927224],
       [ 2.61173377, -0.40054523, -0.3690041 ],
       [ 1.10489233, -1.3502401 , -0.33534969],
       [ 0.40562784,  1.6345255 , -0.62287599],
       [ 3.01454072, -0.09388323, -0.43408737],
       [ 2.13597933, -1.53793093,  1.59821161],
       [ 0.59330429,  1.32307389, -1.04905444],
       [ 0.65048203, -1.08185649, -0.36457733],
       [ 0.3702715 ,  1.69262476, -0.68328002],
       [-1.13816744, -0.00937671, -0.64013047],
       [ 1.78842516, -1.54995222,  0.77821554],
       [ 0.8099707 ,  1.36182829, -1.47551252],
       [ 0.06052699, -0.5740107 ,  0.44690002],
       [-0.40241265, -0.25016624,  0.17326464]]

        var test_examples = [[ 0.56645301,  1.1245596 , -0.43972837],
       [ 1.2918756 , -0.63501136,  3.50284953],
       [-1.02497955,  2.4921979 , -0.75041686],
       [ 2.01112297, -1.50679903,  1.75976501]]

        var train_labels = [1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1]
        var test_labels = [1, -1, 1, -1, -1]
        
        for(var i = 0; i < train_examples.length; i++) {
          randomforest.addExample(train_examples[i], train_labels[i]);
        }

        randomforest.train();
        for(var i = 0; i < test_examples.length; i++) {
          expect(randomforest.classify(test_examples[i])).toBe(test_labels[i])
        }
        

        // random forest are not deterministic, check on average it works
        var count = 0;
        for(var tests=0; tests<100; tests++){
            if(randomforest.classify([ 1.76931368, -1.15994557,  2.32671689]) == -1) {
                count++;
            }
        }
        expect(count).toBeGreaterThan(50);
    }); 

    it('should perform multiclass classification with 3d features', function() {
        var randomforest = new RandomForestClassifier();

        // generated from sklearn.datasets.make_classification
        var train_examples = [
            [ 0.8080942 ,  1.30614285, -1.35468264],
            [ 2.77361167, -0.72931586, -1.29065071],
            [-0.49200143,  1.24270065,  2.13055164],
            [ 4.66522504, -1.04715041, -0.50329974],
            [-1.8997677 ,  1.06832484, -0.71905057],
            [-0.97324051, -0.53861337,  0.83167627],
            [-0.89175073, -0.8998025 ,  0.9771675 ],
            [-0.71374886,  1.95640935,  0.21539975],
            [-1.13796921, -1.17133199,  1.09793521],
            [-0.73763646,  4.46314491,  3.14618865],
            [-1.78924113,  0.82935002, -0.74527505],
            [-0.97522451, -1.56499373,  1.12782555],
            [ 1.80333287,  0.77424192, -1.52114292],
            [-1.05755709, -0.37600237, -1.16263587],
            [ 1.20867444,  3.60765148, -0.23910699]]

        var test_examples = [
            [-1.81077556,  0.34670694,  0.46382542],
            [ 0.8668284 ,  2.67468251, -0.08890863],
            [ 0.75710073,  1.64514855, -1.49304881],
            [-0.08449747,  1.05468084,  2.74085508],
            [-1.08888363, -1.94654662,  0.42889803]]

        var train_labels = [0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 1, 0, 1, 0]
        var test_labels = [2, 0, 0, 2, 1]

        for(var i = 0; i < train_examples.length; i++) {
          randomforest.addExample(train_examples[i], train_labels[i]);
        }

        randomforest.train();
        for(var i = 0; i < test_examples.length; i++) {
          expect(randomforest.classify(test_examples[i])).toBe(test_labels[i])
        }

        // random forest are not deterministic, check on average it works
        var count = 0;
        for(var tests=0; tests<100; tests++){
            randomforest.train();
            if(randomforest.classify([1.0, 2.0, 3.0]) == 0) {
                count++;
            }
        }
        expect(count).toBeGreaterThan(50);
    });

  it('should perform multiclass classification with 3d features & 2D (dot product) decisions', function() {
        var randomforest = new RandomForestClassifier({type:1});

        // generated from sklearn.datasets.make_classification
        var train_examples = [
            [ 0.8080942 ,  1.30614285, -1.35468264],
            [ 2.77361167, -0.72931586, -1.29065071],
            [-0.49200143,  1.24270065,  2.13055164],
            [ 4.66522504, -1.04715041, -0.50329974],
            [-1.8997677 ,  1.06832484, -0.71905057],
            [-0.97324051, -0.53861337,  0.83167627],
            [-0.89175073, -0.8998025 ,  0.9771675 ],
            [-0.71374886,  1.95640935,  0.21539975],
            [-1.13796921, -1.17133199,  1.09793521],
            [-0.73763646,  4.46314491,  3.14618865],
            [-1.78924113,  0.82935002, -0.74527505],
            [-0.97522451, -1.56499373,  1.12782555],
            [ 1.80333287,  0.77424192, -1.52114292],
            [-1.05755709, -0.37600237, -1.16263587],
            [ 1.20867444,  3.60765148, -0.23910699]]

        var test_examples = [
            [-1.81077556,  0.34670694,  0.46382542],
            [ 0.8668284 ,  2.67468251, -0.08890863],
            [ 0.75710073,  1.64514855, -1.49304881],
            [-0.08449747,  1.05468084,  2.74085508],
            [-1.08888363, -1.94654662,  0.42889803]]

        var train_labels = [0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 1, 0, 1, 0]
        var test_labels = [2, 0, 0, 2, 1]

        for(var i = 0; i < train_examples.length; i++) {
            randomforest.addExample(train_examples[i], train_labels[i]);
        }

        randomforest.train();
        // the weak learner is especially non-deterministic as the tree is trained on the dot product of 2 randomly selected dimensions
        var count = 0;
        for(var tests=0; tests<200; tests++){
            randomforest.train();
            if(randomforest.classify([1.0, 2.0, 3.0]) == 0) {
                count++;
            }
        }
        expect(count).toBeGreaterThan(50);
		}); 
});
