package errmsg

const (
	SUCCESS = 200
	ERROR   = 500

	//用户模块需要捕捉的错误

	ERROR_USERNAME_USED    = 1001
	ERROR_PASSWORD_WRONG   = 1002
	ERROR_USER_NOT_EXIST   = 1003
	ERROR_TOKEN_EXIST      = 1004
	ERROR_TOKEN_RUNTIME    = 1005
	ERROR_TOKEN_WRONG      = 1006
	ERROR_TOKEN_TYPE_WRONG = 1007
	ERROR_USER_NO_RIGHT    = 1008

	//分区模块的错误
	ERROR_CATEGORY_USED     = 2001
	ERROR_CATEGORY_NOTEXIST = 2002

	//文章模块的错误
	ERROR_ARTICLE_NOTFOUND = 3001
)

var CodeMsg = map[int]string{
	SUCCESS:                 "OK",
	ERROR:                   "FAIL",
	ERROR_USERNAME_USED:     "用户名已存在",
	ERROR_PASSWORD_WRONG:    "密码错误",
	ERROR_USER_NOT_EXIST:    "用户名不存在",
	ERROR_TOKEN_EXIST:       "TOKEN不存在",
	ERROR_TOKEN_RUNTIME:     "TOKEN已过期",
	ERROR_TOKEN_WRONG:       "TOKEN验证错误",
	ERROR_TOKEN_TYPE_WRONG:  "TOKEN格式错误",
	ERROR_CATEGORY_NOTEXIST: "分区模块不存在",
	ERROR_CATEGORY_USED:     "分区模块以存在",
	ERROR_ARTICLE_NOTFOUND:  "文章未找到",
}

func GetErrorMsg(code int) string {
	return CodeMsg[code]
}
