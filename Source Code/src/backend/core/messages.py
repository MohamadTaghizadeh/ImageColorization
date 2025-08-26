class Message:
    """
    params: language: str = 'en' | 'fa'. default: 'en'
    response: {'status': bool, 'code': str, 'message': str}
    Example of usage:
        from base_fastapi_utils.api.core.messages import messages
        messages('en').ERR_MORE_THAN_N_TRY(config.MAX_NUMBER_OF_TRY, cellnum, status: bool (default=false))
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self.code_msg = {
            "E100": {
                "fa": "خطا در اتصال به سرور {server_name}",
                "en": "Failed to connect to {server_name} server",
            },
            "E101": {
                "fa": "خطا در اتصال به پایگاه داده",
                "en": "Failed to connect to database",
            },
            "E102": {"fa": "لطفا دوباره تلاش کنید", "en": "Please try again"},
            "E103": {
                "fa": "لطفاً دقایقی دیگر مجددا تلاش نمایید",
                "en": "Please try again later",
            },
            "E104": {"fa": "پایگاه داده قفل است", "en": "Database locked"},
            "E105": {
                "fa": "نام کاربری یا کلمه عبور اشتباه است",
                "en": "Username or password is wrong",
            },
            "E106": {
                "fa": "پارامتر {parameter} صحیح نیست",
                "en": "{parameter} parameter is wrong",
            },
            "E107": {"fa": "{msg}", "en": "{msg}"},
            "E108": {"fa": "دسترسی غیر مجاز", "en": "Unauthorized access"},
            "E109": {"fa": "{username} یافت نشد", "en": "{username} not found"},
            "E110": {
                "fa": "{username} اشتباه است یا وجود ندارد",
                "en": "{username} is wrong or not exist",
            },
            "E111": {
                "fa": "{username} اضافه نشد. از قبل موجود است",
                "en": "Failed to add {username}. Already exist",
            },
            "E112": {
                "fa": "{username} خطا در بروزرسانی",
                "en": "Failed to update {username}",
            },
            "E113": {"fa": "{username} حذف نشد", "en": "Failed to delete {username}"},
            "E114": {
                "fa": "بیش از {count} تلاش برای {username}",
                "en": "More than {count} try for {username}",
            },
            "E115": {"fa": "{username} فعال نیست", "en": "{username} is not enabled"},
            "E116": {
                "fa": "{username} تایید نشده است",
                "en": "{username} is not verified",
            },
            "E117": {
                "fa": "حساب {username} منقضی شده است",
                "en": "{username} account has expired",
            },
            "E118": {
                "fa": "حساب {username} هنوز فعال نشده است",
                "en": "{username} account has not started yet",
            },
            "E119": {
                "fa": "درخواست کد برای {username} ثبت نشده یا منقضی شده است",
                "en": "No OTP request exist for {username} or Timeout",
            },
            "E120": {"fa": "{value} اشتباه است", "en": "{value} is wrong"},
            "E121": {
                "fa": "شماره {cellnum} با کد {otp} مطابقت ندارد",
                "en": "{cellnum} doesn't match with code {otp}",
            },
            "E122": {
                "fa": "خطا در درج در پایگاه داده",
                "en": "Failed to insert to DB",
            },
            "E123": {
                "fa": "ورودی نامعتبر",
                "en": "Invalid input",
            },
            "E124": {
                "fa": "عدم تطابق کد ملی و تاریخ تولد",
                "en": "National code and birth date doesn't match",
            },
            "E125": {
                "fa": "خطا در دریافت تصویر ثبت احوال",
                "en": "Failed to get national card image",
            },
            "E126": {
                "fa": "تعداد تلاش مجدد بیش از حد مجاز",
                "en": "Number of retries exceeded",
            },
            "E127": {
                "fa": "احراز هویت قبلا انجام شده است",
                "en": "Authentication has already been done",
            },
            "E128": {
                "fa": "ناموفق",
                "en": "Failed",
            },
            "E129": {
                "fa": "سرویس ثبت احوال در دسترس نیست",
                "en": "SABT-AHVAL service unavailable",
            },
            "E130": {
                "fa": "خطا در سرویس ثبت احوال",
                "en": "SABT-AHVAL service failed",
            },
            "E131": {
                "fa": "تصویر ثبت احوال یافت نشد",
                "en": "SABT-AHVAL response has no image",
            },
            "E132": {
                "fa": "شناسه نامعتبر",
                "en": "ID is not valid",
            },
            "E133": {
                "fa": "سرویس در دسترس نیست، لطفا لحظاتی دیگر مجددا تلاش نمایید",
                "en": "Service is unavailable, please try again in a few moments",
            },
            "E134": {
                "fa": "شناسه درخواست تکراری است",
                "en": "Request ID is duplicate",
            },
            "S100": {"fa": "موفق", "en": "Success"},
        }

    def get_msg_from_code(self, code: str, **kwargs):
        return self.code_msg[code][self.language].format(**kwargs)

    def ERR_FAILED_TO_CONNECT_TO_SERVER(self, server_name: str, status: bool = False):
        code = "E100"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, server_name=server_name),
        }

    def ERR_FAILED_TO_CONNECT_TO_DATABASE(self, status: bool = False):
        code = "E101"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_TRY_AGAIN(self, status: bool = False):
        code = "E102"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_TRY_AGAIN_LATER(self, status: bool = False):
        code = "E103"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_DB_LOCKED(self, status: bool = False):
        code = "E104"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_USERNAME_OR_PASSWORD_IS_WRONG(self, status: bool = False):
        code = "E105"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_WRONG_PARAMS(self, parameter: str, status: bool = False):
        code = "E106"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, parameter=parameter),
        }

    def ERR_CUSTOM(self, msg: str, status: bool = False):
        code = "E107"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, msg=msg),
        }

    def ERR_UNAUTHORIZED_ACCESS(self, status: bool = False):
        code = "E108"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_NOT_FOUND(self, username: str, status: bool = False):
        code = "E109"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_WRONG_OR_NOT_EXIST(self, username: str, status: bool = False):
        code = "E110"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_FAILED_TO_CREATE(self, username: str, status: bool = False):
        code = "E111"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_FAILED_TO_UPDATE(self, username: str, status: bool = False):
        code = "E112"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_FAILED_TO_DELETE(self, username: str, status: bool = False):
        code = "E113"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_MORE_THAN_N_TRY(self, count: int, username: str, status: bool = False):
        code = "E114"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(
                code=code, username=username, count=count
            ),
        }

    def ERR_NOT_ENABLE(self, username: str, status: bool = False):
        code = "E115"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_NOT_VERIFIED(self, username: str, status: bool = False):
        code = "E116"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_EXPIRED(self, username: str, status: bool = False):
        code = "E117"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_NOT_STARTED(self, username: str, status: bool = False):
        code = "E118"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_NO_RECORD_IN_OTP_OR_TIMEOUT(self, username: str, status: bool = False):
        code = "E119"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, username=username),
        }

    def ERR_WRONG_CODE(self, value: str, status: bool = False):
        code = "E120"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, value=value),
        }

    def ERR_NO_CELLNUM_WITH_OTP(self, cellnum: str, otp: str, status: bool = False):
        code = "E121"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code, cellnum=cellnum, otp=otp),
        }

    def ERR_FAILED_TO_ADD_TO_DB(self, status: bool = False):
        code = "E122"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_INVALID_INPUT(self, status: bool = False):
        code = "E123"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_NATIONAL_CODE_AND_BIRTH_DATE_DOES_NOT_MATCH(self, status: bool = False):
        code = "E124"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_FAILED_TO_GET_NATIONAL_CARD_IMAGE(self, status: bool = False):
        code = "E125"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_MAX_RETRY_EXCEEDED(self, status: bool = False):
        code = "E126"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_AUTHENTICATION_HAS_ALREADY_BEEN_DONE(self, status: bool = False):
        code = "E127"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_FAILED(self, status: bool = False):
        code = "E128"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_SABT_AHVAL_SERVICE_UNAVAILABLE(self, status: bool = False):
        code = "E129"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_SABT_AHVAL_SERVICE_FAILED(self, status: bool = False):
        code = "E130"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_SABT_AHVAL_RESPONSE_HAS_NO_IMAGE(self, status: bool = False):
        code = "E131"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_INVALID_ID(self, status: bool = False):
        code = "E132"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def ERR_SERVICE_UNAVAILABLE(self, status: bool = False):
        code = "E133"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }
    def ERR_DUPLICATE_REQUEST_ID(self, status: bool = False):
        code = "E134"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }

    def INF_SUCCESS(self, status: bool = True):
        code = "S100"
        return {
            "status": status,
            "code": code,
            "message": self.get_msg_from_code(code=code),
        }
