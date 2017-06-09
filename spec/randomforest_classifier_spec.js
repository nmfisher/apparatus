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

describe('randomforest', function() {
    it('should perform binary classification with 2d features', function() {
        var randomforest = new RandomForestClassifier();

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

        randomforest.train();

        expect(randomforest.classify([-0.5 , -0.5 ])).toBe(1);

        // random forest are not deterministic, check on average it works
        var count = 0;
        for(var tests=0; tests<100; tests++){
            randomforest.train();
            if(randomforest.classify([1.0, 2.0]) == 1) {
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
