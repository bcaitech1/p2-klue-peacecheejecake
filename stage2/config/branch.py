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


    # def __str__(self):
    #     str_ = ""
    #     for name, value in self.__dict__.items():
    #         if value:
    #             str_ += f".{name}={value}, "
        
    #     return str_[:-2]


    def assign(self, **kwarg):
        for name, attr in kwarg.items():
            if type(attr) == dict:
                attr = ConfigBranch(**attr)
            setattr(self, name, attr)
        return kwarg
