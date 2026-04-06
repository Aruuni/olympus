#include <linux/module.h>
#include <linux/export-internal.h>
#include <linux/compiler.h>

MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x8bddd836, "__nlmsg_put" },
	{ 0x9166fada, "strncpy" },
	{ 0x74a885b2, "netlink_unicast" },
	{ 0x656e4a6e, "snprintf" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x9eb487db, "tcp_yeah" },
	{ 0xd513aa4b, "tcp_astraea_ops" },
	{ 0x1c158d7b, "tcp_illinois" },
	{ 0x27a00a3a, "tcp_bbr1_cong_ops" },
	{ 0x89686af5, "tcp_veno" },
	{ 0xe3494b3a, "htcp" },
	{ 0x2817b56c, "tcp_bbr3_cong_ops" },
	{ 0x653ceb64, "tcp_westwood" },
	{ 0x39116b8b, "tcp_vegas" },
	{ 0xfc5077a0, "bic" },
	{ 0x43ab3c6e, "tcp_cdg" },
	{ 0xb5c2dda8, "tcp_highspeed" },
	{ 0x30245bb8, "tcp_hybla" },
	{ 0x54b1fac6, "__ubsan_handle_load_invalid_value" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x6383b27c, "__x86_indirect_thunk_rdx" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x65487097, "__x86_indirect_thunk_rax" },
	{ 0xf90a1e85, "__x86_indirect_thunk_r8" },
	{ 0x122c3a7e, "_printk" },
	{ 0x37a0cba, "kfree" },
	{ 0xc9fc5662, "netlink_kernel_release" },
	{ 0x3ac96021, "tcp_unregister_congestion_control" },
	{ 0x4c03a563, "random_kmalloc_seed" },
	{ 0x72b29fc2, "kmalloc_caches" },
	{ 0xf9607962, "__kmalloc_cache_noprof" },
	{ 0x8f8d1514, "init_net" },
	{ 0x239a4f2e, "__netlink_kernel_create" },
	{ 0x8ff57cf2, "cubictcp" },
	{ 0xb7e4a3fe, "tcp_register_congestion_control" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x754d539c, "strlen" },
	{ 0x553e91c0, "__alloc_skb" },
	{ 0x2b91227d, "module_layout" },
};

MODULE_INFO(depends, "tcp_yeah,tcp_astraea,tcp_illinois,tcp_veno,tcp_htcp,tcp_bbr,tcp_westwood,tcp_vegas,tcp_bic,tcp_cdg,tcp_highspeed,tcp_hybla");


MODULE_INFO(srcversion, "4F816DFB8089463E28EFB1A");
