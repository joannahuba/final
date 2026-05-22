from abc import ABC

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