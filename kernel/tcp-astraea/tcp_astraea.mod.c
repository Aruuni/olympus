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

KSYMTAB_DATA(tcp_astraea_ops, "_gpl", "");

SYMBOL_CRC(tcp_astraea_ops, 0xd513aa4b, "_gpl");

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x46259bff, "tcp_mss_to_mtu" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x122c3a7e, "_printk" },
	{ 0xb7e4a3fe, "tcp_register_congestion_control" },
	{ 0x3ac96021, "tcp_unregister_congestion_control" },
	{ 0x2b91227d, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "EEBD720050E9AF9E5B2B86A");
