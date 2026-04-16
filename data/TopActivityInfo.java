package com.jd.m.news.soa.jsf.domain.topActivity.activities;

import com.jd.m.news.soa.jsf.domain.topActivity.activityDetails.LotteryDetails;
import com.jd.m.news.soa.jsf.domain.topActivity.activityDetails.TopActShopInfo;
import lombok.Data;

import java.sql.Timestamp;
import java.util.Date;
import java.util.List;

@Data
public class TopActivityInfo {


    /**
     * 配置id
     */
    private Integer id;

    /**
     * Banner图片
     */
    private String image;

    /**
     * banner名称
     */
    private String name;

    /**
     * 利益点标签图
     */
    private String benefitsTag;

    /**
     * 利益点文案
     */
    private String benefitsText;

    /**
     * 原价
     */
    private String price;

    /**
     * 抽签价
     */
    private String lotteryPrice;

    /**
     * 库存件数
     */
    private Integer stockNum;

    /**
     * 活动状态, 0表示未开始，1表示进行中，-1表示已结束 (该字段不以库中数据为准)
     */
    private Integer status;

    /**
     * 活动开始时间
     */
    private Date startTime;

    /**
     * 活动结束时间
     */
    private Date endTime;

    /**
     * 开奖时间(公布时间)
     */
    private Date drawTime;

    /**
     * 活动抽签类型，1：1元抽， 2:原价抽 - 对应后台的activityType
     */
    private Integer lotteryType;

    /**
     * 头像列表
     */
    private List<String> avatars;

    /**
     * 参与人数
     */
    private Long participantCount;

    /**
     * 中奖人数 - 只有往期活动有
     */
    private Integer winnerCount;

    /**
     * 产品特性标签，0没有，1自营，后续增加枚举管理
     */
    private Integer commercialLabel;

    /**
     * skuId
     */
    private String skuId;

    /**
     * 是否中奖 0用户没中奖，1用户中奖了
     */
    private Integer winner;

    private LotteryDetails lotteryDetails;
    /**
     * 抽签场景：1、定时开奖，2、即抽即玩（默认定时开奖）
     */
    private Integer activityScene;
    /**
     * 中奖时间 - 即买即抽专有，最近一次中签的够买截止时间
     */
    private Timestamp winTime;
    /**
     * 购买截止时间 - 即买即抽专有，最近一次中签的够买截止时间
     */
    private Timestamp purchaseEndTime;
    /**
     * 活动排序号
     */
    private Integer sortNum;
    /**
     * 尖货商家信息
     */
    private TopActShopInfo shopInfo;
    /**
     * 开始前已解锁人数
     */
    private Integer unlockCount;
    /**
     * 解锁类型 0、空：不用解锁 1：开始前解锁 2：开始后解锁
     */
    private Integer unlockType;
    /**
     * 用户解锁状态
     */
    private Boolean userUnlockStatus;
    /**
     * 剩余库存
     */
    private Integer leftNum;
}
