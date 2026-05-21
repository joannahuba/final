from abc import ABC, abstractmethod




class SequencePredictorModelWrapper:

    def __init__(self, models_dict: dict, optimizers_list: list, kmer_sequences: list = None, validation_config: dict):
        """
        models_dict[model_path, model_loader_path] - słownik z modelami 
        optimizers_list - lista zdefiniowanych obiektów
        """
        self.models_dict = models_dict
        self.optimizers_list = optimizers_list
        self.kmer_sequences = kmer_sequences
        #TODO - check may errors exists 
        self.validation_config = self.SetValidationFunction(validation_config)
    

    def OptimizeSequence(self, sequences: list):
        """
        methody są z zadań (rekonstrukcja  | optymalizacja)

        zwraca dict (sekwencja lub coś innego) : przewidzianymi sekwencjami z  `GeneralSequenceOptimizer`

        method: odpoaida 

        sequences lista słowników sekwencja: [metoda, config_rekonstrukcji] 
        """

        # do każdej metody podajemy sekwencje 

        # każda metoda nam wypluwa do sekwencji propozycje z ich przewidzianą jakością

        # na każdą sekwencje agregujemy jaki jest
        
        # funkcja kosztu po wszystkich predykcjach (min z predykcji + ewentualnie warunek na duży outliery) 
        
        # zwróćenie sekwencji sekwencja | minimum po wszystkim | score'y poszczególnych optimizerów | optim_name_iter_i
        # 
        #  

        pass


    # setters & getters

    def SetValidationFunction(self):
        """
        return callable responsible for sequence check: 
        ta metoda tworzy funkcje callable 

        funkcja
        przyjmuje sekwencje wektorowo

        zwraca boolean liste które warunki spełniają warunki 
        """
        pass

    # helpers

    # def EvaluateModels(self, sequences):
    #     """
    #     dajemy sekwencje i funkcja zwraca wyniki z predykcji 

    #     lista sekwencji i do każdej informacja o tym jak bardzo zmieniona jest ekspresja 
    #     """



class GeneralSequenceOptimizer(ABC):

    @abstractmethod
    def __init__(self, kmer):
        super().__init__()

    
    @abstractmethod
    def __call__(self, sequence: str, method: str, models_list, validation_function: callable, reconstruction_config: dict | None = None):
        """
        przyjmuje te metody i skewncje i walidacyjn, zależnie od methody będziemy albo rekonstruować (inaczej trzeba wymagania) 
        
        """

        # dobranie konfiguracji tych k-merów | ilości sekwencji do optymalizacji 
        ## zależnie od metody mamy inne ustawienia 

        # optymalizacja zwraca sekwencje

        # walidacja - które są sensowne 
        ## czyli sprawdzamy te predykcje o ile się zmieniają
        ## odrzucamy te które nie spełniają warunków 

        # zwracamy nowa sekwencja | historia zmian sekwencji 





        pass


    @abstractmethod
    def SpecifyMethod(method):
        """
        TODO FUnkcja odpowiedzialna za definowane tej struktury jak dobirerać ilośc itd. 
        """
        pass 


    @abstractmethod
    def Optimization(self):
        """
        Abstract method responsible for generating new sequences 
        """
        pass

    @abstractmethod
    def Validation(self, method):
        """
        Funkcja odpowiedzialna za walidacje 
        """
        pass