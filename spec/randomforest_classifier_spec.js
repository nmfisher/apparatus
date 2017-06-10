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
        for(var tests=0; tests<200; tests++){
            randomforest.train();
            if(randomforest.classify([1.0, 2.0]) == -1) {
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
        var correct = 0;
        for(var i = 0; i < test_examples.length; i++) {

            if(randomforest.classify(test_examples[i]) == test_labels[i]) {
                correct++;
            }
        }
        
        expect(correct).toBeGreaterThan(test_examples.length * 0.9); 

    });
    
});
