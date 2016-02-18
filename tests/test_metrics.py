from unittest import TestCase
from sklearn_evaluation.metrics import precision_at

class Test_Precision_At(TestCase):
    def test_perfect_precision(self):
        labels = [1  ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,0 ,0]
        scores = [100,90,80,70,60,50,40,30,20,10]
        prec, cutoff = precision_at(labels, scores, 0.10)
        self.assertEqual(prec, 1.0)
        self.assertEqual(cutoff, 100)
    def test_baseline_precision(self):
        labels = [1  ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,0 ,0]
        scores = [100,90,80,70,60,50,40,30,20,10]
        prec, cutoff = precision_at(labels, scores, 1.0)
        self.assertEqual(prec, 0.5)
        self.assertEqual(cutoff, 10)