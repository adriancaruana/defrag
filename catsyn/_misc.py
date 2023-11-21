from functools import lru_cache
import hashlib

import numpy as np


def str_to_state(s: str) -> int:
    """Generate an integer from unique attributes of this object. Must be
    between 0 and 2**32 - 1.

    """
    # return np.random.RandomState(
    #     int.from_bytes(bytes(s, "utf-8"), "little") % (2 ** 32 - 1)
    # )
    return np.random.default_rng(
        int.from_bytes(bytes(s, "utf-8"), "little") % (2 ** 32 - 1)
    )


def md5_hexdigest(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()


class PhonemicName:
    _prefixes = (
        "dozmarbinwansamlitsighidfidlissogdirwacsabwissib"
        "rigsoldopmodfoglidhopdardorlorhodfolrintogsilmir"
        "holpaslacrovlivdalsatlibtabhanticpidtorbolfosdot"
        "losdilforpilramtirwintadbicdifrocwidbisdasmidlop"
        "rilnardapmolsanlocnovsitnidtipsicropwitnatpanmin"
        "ritpodmottamtolsavposnapnopsomfinfonbanmorworsip"
        "ronnorbotwicsocwatdolmagpicdavbidbaltimtasmallig"
        "sivtagpadsaldivdactansidfabtarmonranniswolmispal"
        "lasdismaprabtobrollatlonnodnavfignomnibpagsopral"
        "bilhaddocridmocpacravripfaltodtiltinhapmicfanpat"
        "taclabmogsimsonpinlomrictapfirhasbosbatpochactid"
        "havsaplindibhosdabbitbarracparloddosbortochilmac"
        "tomdigfilfasmithobharmighinradmashalraglagfadtop"
        "mophabnilnosmilfopfamdatnoldinhatnacrisfotribhoc"
        "nimlarfitwalrapsarnalmoslandondanladdovrivbacpol"
        "laptalpitnambonrostonfodponsovnocsorlavmatmipfip"
    )
    _suffixes = (
        "zodnecbudwessevpersutletfulpensytdurwepserwylsun"
        "rypsyxdyrnuphebpeglupdepdysputlughecryttyvsydnex"
        "lunmeplutseppesdelsulpedtemledtulmetwenbynhexfeb"
        "pyldulhetmevruttylwydtepbesdexsefwycburderneppur"
        "rysrebdennutsubpetrulsynregtydsupsemwynrecmegnet"
        "secmulnymtevwebsummutnyxrextebfushepbenmuswyxsym"
        "selrucdecwexsyrwetdylmynmesdetbetbeltuxtugmyrpel"
        "syptermebsetdutdegtexsurfeltudnuxruxrenwytnubmed"
        "lytdusnebrumtynseglyxpunresredfunrevrefmectedrus"
        "bexlebduxrynnumpyxrygryxfeptyrtustyclegnemfermer"
        "tenlusnussyltecmexpubrymtucfyllepdebbermughuttun"
        "bylsudpemdevlurdefbusbeprunmelpexdytbyttyplevmyl"
        "wedducfurfexnulluclennerlexrupnedlecrydlydfenwel"
        "nydhusrelrudneshesfetdesretdunlernyrsebhulryllud"
        "remlysfynwerrycsugnysnyllyndyndemluxfedsedbecmun"
        "lyrtesmudnytbyrsenwegfyrmurtelreptegpecnelnevfes"
    )

    @classmethod
    @lru_cache()
    def split(cls, s: str, groupby: int = 3):
        if len(s) > groupby:
            return [s[:groupby]] + cls.split(s[groupby:])
        return [s]

    @classmethod
    def new_name(cls, state: np.random.RandomState = np.random) -> str:
        return (
            f"{state.choice(cls.split(cls._prefixes))}"
            f"{state.choice(cls.split(cls._suffixes))}-"
            f"{state.choice(cls.split(cls._prefixes))}"
            f"{state.choice(cls.split(cls._suffixes))}"
        )
