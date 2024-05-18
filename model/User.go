package model

import (
	"blog/utils"
	"blog/utils/errmsg"
	"github.com/jinzhu/gorm"
)

// User 声明结构体
type User struct {
	gorm.Model
	Username string `gorm:"type: varchar(20); not null" json:"username"`
	Password string `gorm:"type: varchar(20); not null" json:"password"`
	Role     int    `gorm:"type: int" json:"role"`
}

// CheckUser 查询用户是否存在
func CheckUser(name string) int {
	var users User
	db.Select("id").Where("username = ?", name).First(&users)

	if users.ID > 0 {
		return errmsg.ERROR_USERNAME_USED // code:1001
	}
	return errmsg.SUCCESS
}

// CreateUser 新增用户
func CreateUser(data *User) int {
	//data.Password = utils.ScryptPw(data.Password)
	data.BeforeSave()
	err := db.Create(&data).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}

// GetUsers 查询用户列表
func GetUsers(pageSize int, pageNum int) []User {
	var users []User
	//使用分页将输出列表分隔
	//固定写法
	err := db.Model(&User{}).Limit(pageSize).
		Offset((pageNum - 1) * pageSize).Find(&users).Error
	if err != nil {
		return nil
	}
	return users
}

// EditUser 编辑用户
func EditUser(id int, data *User) int {
	//var user User
	var maps = make(map[string]interface{})
	maps["username"] = data.Username
	maps["role"] = data.Role
	err := db.Model(&User{}).Where("id = ?", id).Updates(maps).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}

// DeleteUser 删除用户
func DeleteUser(id int) int {
	var user User
	err := db.Where("id = ?", id).Delete(&user).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}

// BeforeSave 钩子函数
func (u *User) BeforeSave() {
	u.Password = utils.ScryptPw(u.Password)
}

// CheckLogin 登录验证
func CheckLogin(username string, password string) int {
	var user User
	db.Where("username = ?", username).First(&user)

	//用户是否存在
	if user.ID == 0 {
		return errmsg.ERROR_USER_NOT_EXIST

	}

	//用户密码是否正确
	if user.Password != utils.ScryptPw(password) {
		return errmsg.ERROR_PASSWORD_WRONG
	}

	//用户是否有管理权限
	if user.Role != 0 {
		return errmsg.ERROR_USER_NO_RIGHT
	}

	return errmsg.SUCCESS
}
