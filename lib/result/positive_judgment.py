
class Judgment:
    def __init__(self, guss_malicious, fact_malicious):
        self.fact = fact_malicious
        self.guss = guss_malicious
        self.TP = 0
        self.TN = 0
        self.all_T_numbers = 1e-10
        self.all_F_numbers = 1e-10
        self.TPR = 0
        self.TNR = 0

    def compare(self):
        for index, item in enumerate(self.guss):
            if item == True:
                self.all_T_numbers += 1
                if item == self.fact[index]:
                    self.TP += 1
            else:
                self.all_F_numbers += 1
                if item == self.fact[index]:
                    self.TN += 1
        self.TPR = self.TP / self.all_T_numbers
        self.TNR = self.TN / self.all_F_numbers
        return self.TPR, self.TNR
