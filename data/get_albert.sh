#!/usr/bin/env bash
# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#

wget https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz

tar -xvzf albert_base_v2.tar.gz
rm albert_base_v2.tar.gz
mv albert_base albert_pretrain_base

wget https://storage.googleapis.com/albert_models/albert_large_v2.tar.gz
tar -xvzf albert_large_v2.tar.gz
rm albert_large_v2.tar.gz
mv albert_large albert_pretrain_large