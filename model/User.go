package model

import (
	"blog/utils"
	"blog/utils/errmsg"
	"github.com/jinzhu/gorm"
)

type User struct {
	gorm.Model
	Username string `gorm:"type: varchar(20); not null" json:"username" validate:"required,min=4,max=12" label:"用户名"`
	Password string `gorm:"type: varchar(20); not null" json:"password" validate:"required,min=6,max=20" label:"密码"`
	Role     int    `gorm:"type: int;default:2" json:"role" validate:"required,gte=2" label:"角色"`
}

func CheckUser(name string) int {
	var users User
	db.Select("id").Where("username = ?", name).First(&users)

	if users.ID > 0 {
		return errmsg.ERROR_USERNAME_USED // code:1001
	}
	return errmsg.SUCCESS
}
func CreateUser(data *User) int {
	//data.Password = utils.ScryptPw(data.Password)
	data.BeforeSave()
	err := db.Create(&data).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}
func GetUsers(pageSize int, pageNum int) ([]User, int64) {
	var users []User
	var total int64
	//使用分页将输出列表分隔
	//固定写法
	err := db.Model(&User{}).Limit(pageSize).
		Offset((pageNum - 1) * pageSize).Find(&users).Count(&total).Error
	if err != nil {
		return nil, 0
	}
	return users, total
}
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
func DeleteUser(id int) int {
	var user User
	err := db.Where("id = ?", id).Delete(&user).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}
func (u *User) BeforeSave() {
	u.Password = utils.ScryptPw(u.Password)
}
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
