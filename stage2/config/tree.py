##########################################
# CONFIGURATION ##########################
##########################################


class ConfigBranch(dict):
    def __init__(self, **kwarg):
        self.__dict__ = self
        self.assign(**kwarg)
            
    
    def __call__(self, **kwarg):
        self.assign(**kwarg)
        return self.__dict__


    def __bool__(self):
        return any(self.__dict__.values())


    def __str__(self):
        str_ = ""
        for name, value in self.__dict__.items():
            if value:
                str_ += f".{name}={value}, "
        
        return str_[:-2]


    def assign(self, **kwarg):
        for name, attr in kwarg.items():
            setattr(self, name, attr)
        return kwarg
    
    

class ConfigTree:
    def __init__(self):
        self.system = ConfigBranch()
        self.path = ConfigBranch()
        self.data = ConfigBranch()
        self.model = ConfigBranch()
        self.train = ConfigBranch()
        self.test = ConfigBranch()

    
    def __bool__(self):
        return any(self.__dict__.values())


    def __str__(self):
        str_ = ""
        for name, attr in self.__dict__.items():
            if name == 'path' or not attr:
                continue

            str_ += f"({name}) {attr}\n"

        return str_[:-1]