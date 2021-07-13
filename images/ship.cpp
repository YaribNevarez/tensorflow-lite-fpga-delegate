unsigned char ship[] = {
  0x9e, 0xbe, 0xde, 0x9e, 0xbb, 0xda, 0x8b, 0xa6, 0xc2, 0x84, 0x9e, 0xba,
  0xa6, 0xc1, 0xde, 0xb6, 0xd0, 0xec, 0xbb, 0xd3, 0xee, 0xc1, 0xd8, 0xf1,
  0xc7, 0xdb, 0xf3, 0xcd, 0xdd, 0xf5, 0xd1, 0xde, 0xf4, 0xce, 0xda, 0xeb,
  0xda, 0xe5, 0xf0, 0xdf, 0xe8, 0xf1, 0xe3, 0xeb, 0xf2, 0xe6, 0xed, 0xf5,
  0xd5, 0xdc, 0xe3, 0xe2, 0xe9, 0xf0, 0xe7, 0xee, 0xf5, 0xeb, 0xef, 0xf8,
  0xec, 0xf1, 0xf9, 0xe8, 0xef, 0xf3, 0xea, 0xf1, 0xf3, 0xea, 0xf1, 0xf3,
  0xec, 0xf2, 0xf5, 0xe2, 0xe7, 0xeb, 0xe6, 0xeb, 0xef, 0xee, 0xf3, 0xf7,
  0xe8, 0xed, 0xf1, 0xe4, 0xe7, 0xea, 0xed, 0xef, 0xf3, 0xee, 0xf1, 0xf6,
  0xaa, 0xc8, 0xe5, 0xac, 0xc7, 0xe2, 0x97, 0xb0, 0xc9, 0x89, 0xa0, 0xb8,
  0xae, 0xc7, 0xdf, 0xc1, 0xd9, 0xf0, 0xc5, 0xda, 0xf0, 0xc7, 0xda, 0xee,
  0xce, 0xdf, 0xf3, 0xd7, 0xe5, 0xf7, 0xd9, 0xe5, 0xf5, 0xd2, 0xdb, 0xe8,
  0xe1, 0xe9, 0xf3, 0xe7, 0xee, 0xf5, 0xe9, 0xef, 0xf5, 0xed, 0xf3, 0xf8,
  0xdb, 0xe1, 0xe6, 0xe4, 0xea, 0xef, 0xe8, 0xee, 0xf3, 0xf2, 0xf5, 0xfb,
  0xf5, 0xf7, 0xfb, 0xea, 0xed, 0xee, 0xec, 0xef, 0xee, 0xf2, 0xf5, 0xf4,
  0xf1, 0xf5, 0xf8, 0xe4, 0xe9, 0xec, 0xeb, 0xef, 0xf3, 0xf3, 0xf8, 0xfc,
  0xe9, 0xec, 0xf0, 0xe8, 0xe8, 0xec, 0xf6, 0xf6, 0xfa, 0xf6, 0xf7, 0xfb,
  0xae, 0xc9, 0xe1, 0xb0, 0xc8, 0xde, 0x9d, 0xb3, 0xc7, 0x8e, 0xa2, 0xb5,
  0xb5, 0xc9, 0xdb, 0xc9, 0xdc, 0xee, 0xce, 0xdf, 0xef, 0xc7, 0xd6, 0xe4,
  0xd1, 0xdd, 0xeb, 0xdf, 0xe9, 0xf4, 0xda, 0xe2, 0xec, 0xd4, 0xdb, 0xe4,
  0xe0, 0xe5, 0xeb, 0xe6, 0xea, 0xef, 0xe6, 0xea, 0xee, 0xef, 0xf4, 0xf7,
  0xdd, 0xe2, 0xe5, 0xe4, 0xe9, 0xec, 0xe9, 0xee, 0xf1, 0xef, 0xf3, 0xf6,
  0xe8, 0xeb, 0xec, 0xd5, 0xd6, 0xd7, 0xec, 0xea, 0xe8, 0xf3, 0xf1, 0xef,
  0xf5, 0xf7, 0xf8, 0xe7, 0xe9, 0xeb, 0xee, 0xf0, 0xf2, 0xf8, 0xfa, 0xfc,
  0xed, 0xee, 0xf1, 0xe6, 0xe5, 0xe8, 0xfa, 0xf9, 0xfb, 0xf5, 0xf4, 0xf7,
  0xb4, 0xcb, 0xde, 0xb2, 0xc7, 0xd8, 0xa0, 0xb3, 0xc2, 0x93, 0xa4, 0xb3,
  0xba, 0xc9, 0xd6, 0xcb, 0xd9, 0xe4, 0xd4, 0xe1, 0xeb, 0xcf, 0xd9, 0xe1,
  0xd6, 0xdf, 0xe5, 0xe4, 0xeb, 0xef, 0xdd, 0xe3, 0xe6, 0xd6, 0xd9, 0xdd,
  0xdc, 0xdd, 0xe1, 0xe7, 0xe8, 0xec, 0xdf, 0xe2, 0xe6, 0xf0, 0xf4, 0xf7,
  0xe0, 0xe4, 0xe6, 0xe4, 0xe7, 0xea, 0xe9, 0xed, 0xf0, 0xe4, 0xeb, 0xee,
  0xb1, 0xb8, 0xba, 0xac, 0xaf, 0xb0, 0xe6, 0xe4, 0xe5, 0xf3, 0xf1, 0xf1,
  0xf8, 0xf8, 0xf8, 0xe8, 0xe8, 0xe9, 0xee, 0xee, 0xef, 0xfa, 0xfa, 0xfb,
  0xee, 0xed, 0xee, 0xe4, 0xe3, 0xe4, 0xf9, 0xf7, 0xf8, 0xf4, 0xf2, 0xf3,
  0xba, 0xcf, 0xdf, 0xb9, 0xcc, 0xd9, 0xa5, 0xb5, 0xc1, 0x93, 0xa1, 0xac,
  0xbd, 0xc9, 0xd2, 0xcc, 0xd6, 0xdd, 0xd9, 0xe1, 0xe7, 0xcf, 0xd5, 0xd9,
  0xd3, 0xd7, 0xda, 0xe7, 0xed, 0xeb, 0xde, 0xe2, 0xe1, 0xd6, 0xd7, 0xd7,
  0xda, 0xd9, 0xdb, 0xe7, 0xe5, 0xe8, 0xd3, 0xd4, 0xd6, 0xeb, 0xec, 0xee,
  0xe2, 0xe3, 0xe5, 0xe0, 0xe1, 0xe3, 0xe8, 0xec, 0xee, 0xd4, 0xe1, 0xe4,
  0x9f, 0xaa, 0xb0, 0xa8, 0xae, 0xb2, 0xe0, 0xe0, 0xe5, 0xed, 0xec, 0xef,
  0xf7, 0xf7, 0xf7, 0xe7, 0xe7, 0xe7, 0xeb, 0xeb, 0xea, 0xf6, 0xf6, 0xf5,
  0xe8, 0xe7, 0xe7, 0xea, 0xe7, 0xe8, 0xf8, 0xf5, 0xf6, 0xf2, 0xef, 0xf0,
  0xc1, 0xd0, 0xdc, 0xbe, 0xcd, 0xd5, 0xaa, 0xb7, 0xbf, 0x8e, 0x9a, 0xa4,
  0xbf, 0xc9, 0xce, 0xcb, 0xd4, 0xd5, 0xdb, 0xe2, 0xe1, 0xd3, 0xd7, 0xd5,
  0xd7, 0xda, 0xd6, 0xea, 0xee, 0xe6, 0xdd, 0xdf, 0xda, 0xd6, 0xd7, 0xd6,
  0xd6, 0xd7, 0xd6, 0xe4, 0xe5, 0xe3, 0xc7, 0xc7, 0xc8, 0xcd, 0xcd, 0xce,
  0xcf, 0xd1, 0xd0, 0xce, 0xd0, 0xce, 0xeb, 0xef, 0xed, 0xc1, 0xcc, 0xcf,
  0x70, 0x7c, 0x82, 0x9e, 0xa7, 0xad, 0xde, 0xe3, 0xe6, 0xe6, 0xe9, 0xe9,
  0xf5, 0xf6, 0xf4, 0xe5, 0xe5, 0xe1, 0xe2, 0xe1, 0xdb, 0xf1, 0xef, 0xe8,
  0xe4, 0xe2, 0xdd, 0xe7, 0xe6, 0xe7, 0xf3, 0xf1, 0xf2, 0xeb, 0xe8, 0xe6,
  0xc4, 0xcc, 0xd4, 0xbf, 0xc7, 0xca, 0xac, 0xb3, 0xb9, 0x85, 0x8b, 0x96,
  0xbf, 0xc2, 0xc4, 0xca, 0xcb, 0xc9, 0xde, 0xdd, 0xd9, 0xd9, 0xd5, 0xd0,
  0xdf, 0xdb, 0xd3, 0xeb, 0xe9, 0xdb, 0xda, 0xd8, 0xd0, 0xd6, 0xd4, 0xd3,
  0xd7, 0xd8, 0xd4, 0xe3, 0xe6, 0xdd, 0xbc, 0xbc, 0xb9, 0xb0, 0xb0, 0xaf,
  0xbb, 0xbe, 0xbb, 0xba, 0xc0, 0xbd, 0xcd, 0xd4, 0xd2, 0xbb, 0xc0, 0xc5,
  0x78, 0x81, 0x89, 0x89, 0x97, 0x9d, 0xac, 0xb7, 0xb8, 0xb7, 0xbe, 0xbb,
  0xdb, 0xe0, 0xdd, 0xdf, 0xe0, 0xdc, 0xd8, 0xd6, 0xce, 0xeb, 0xe5, 0xda,
  0xe2, 0xdb, 0xd1, 0xe1, 0xde, 0xdb, 0xf0, 0xeb, 0xe7, 0xeb, 0xe0, 0xd8,
  0xcc, 0xcd, 0xd3, 0xc5, 0xc6, 0xc9, 0xae, 0xaf, 0xb3, 0x8c, 0x8a, 0x93,
  0xcb, 0xc0, 0xc5, 0xda, 0xcb, 0xcc, 0xe0, 0xce, 0xce, 0xe0, 0xcc, 0xcb,
  0xe8, 0xd4, 0xd0, 0xed, 0xdf, 0xd2, 0xdc, 0xd0, 0xc6, 0xdc, 0xd2, 0xcd,
  0xdc, 0xd5, 0xce, 0xdd, 0xd8, 0xce, 0xc9, 0xc7, 0xc0, 0xcd, 0xce, 0xcb,
  0xac, 0xb2, 0xb3, 0x8a, 0x95, 0x9b, 0x64, 0x71, 0x7a, 0x53, 0x5d, 0x6a,
  0x47, 0x55, 0x64, 0x3e, 0x50, 0x5e, 0x41, 0x50, 0x58, 0x3c, 0x45, 0x49,
  0x68, 0x6c, 0x6f, 0xb6, 0xb7, 0xb7, 0xd1, 0xce, 0xca, 0xe4, 0xde, 0xd7,
  0xda, 0xd0, 0xc6, 0xd4, 0xc5, 0xb9, 0xef, 0xdd, 0xce, 0xec, 0xd4, 0xc1,
  0xaf, 0xad, 0xb3, 0xaa, 0xa8, 0xad, 0x9d, 0x9a, 0xa0, 0x89, 0x85, 0x8c,
  0xb0, 0xa5, 0xaa, 0xba, 0xac, 0xaf, 0xaf, 0xa0, 0xa1, 0xc5, 0xb3, 0xb3,
  0xd1, 0xbe, 0xbb, 0xd4, 0xc2, 0xb8, 0xce, 0xbd, 0xb4, 0xd2, 0xc3, 0xbd,
  0xd4, 0xc7, 0xc3, 0xc9, 0xbf, 0xb9, 0xc1, 0xc1, 0xba, 0xc1, 0xc6, 0xc4,
  0x8e, 0x98, 0x9f, 0x69, 0x79, 0x89, 0x59, 0x6b, 0x80, 0x5b, 0x6c, 0x80,
  0x54, 0x68, 0x7e, 0x53, 0x68, 0x7e, 0x5e, 0x6e, 0x80, 0x45, 0x4f, 0x5a,
  0x4e, 0x52, 0x58, 0x79, 0x7a, 0x7c, 0xa2, 0x9f, 0x9e, 0xb7, 0xaf, 0xac,
  0xae, 0xa3, 0x9c, 0xa3, 0x95, 0x89, 0xcf, 0xbe, 0xb0, 0xc3, 0xaf, 0x9f,
  0x72, 0x73, 0x7b, 0x73, 0x74, 0x7e, 0x71, 0x72, 0x7b, 0x68, 0x69, 0x70,
  0x69, 0x6d, 0x72, 0x6b, 0x6e, 0x72, 0x6f, 0x70, 0x72, 0x80, 0x7f, 0x80,
  0x8b, 0x87, 0x86, 0x92, 0x88, 0x86, 0x97, 0x8c, 0x89, 0x9b, 0x93, 0x90,
  0x9d, 0x97, 0x98, 0x93, 0x8f, 0x92, 0x97, 0x9e, 0xa0, 0x96, 0xa1, 0xa7,
  0x76, 0x86, 0x93, 0x64, 0x77, 0x8b, 0x63, 0x78, 0x90, 0x63, 0x78, 0x8c,
  0x55, 0x6c, 0x82, 0x56, 0x6c, 0x83, 0x56, 0x66, 0x78, 0x53, 0x5c, 0x68,
  0x8b, 0x8f, 0x95, 0x80, 0x80, 0x83, 0x9a, 0x97, 0x96, 0x99, 0x91, 0x8d,
  0x76, 0x6e, 0x6a, 0x6d, 0x6a, 0x69, 0x84, 0x82, 0x80, 0x7b, 0x78, 0x75,
  0x42, 0x4e, 0x59, 0x4c, 0x57, 0x66, 0x4b, 0x56, 0x62, 0x44, 0x52, 0x58,
  0x53, 0x63, 0x6b, 0x5a, 0x69, 0x73, 0x54, 0x61, 0x68, 0x5a, 0x64, 0x6a,
  0x5d, 0x65, 0x6b, 0x6a, 0x6d, 0x76, 0x66, 0x6a, 0x71, 0x67, 0x6f, 0x73,
  0x6a, 0x72, 0x7b, 0x6b, 0x76, 0x86, 0x72, 0x84, 0x97, 0x6c, 0x7f, 0x92,
  0x5a, 0x6d, 0x80, 0x5a, 0x6d, 0x7f, 0x5b, 0x6e, 0x7f, 0x55, 0x6b, 0x78,
  0x48, 0x60, 0x6f, 0x42, 0x57, 0x68, 0x5f, 0x6c, 0x7a, 0x72, 0x77, 0x80,
  0x80, 0x84, 0x86, 0x6e, 0x6f, 0x6e, 0x93, 0x90, 0x8b, 0xc7, 0xbf, 0xb6,
  0x7d, 0x78, 0x72, 0x67, 0x6f, 0x75, 0x5c, 0x66, 0x6d, 0x5e, 0x68, 0x70,
  0x35, 0x4a, 0x58, 0x41, 0x55, 0x68, 0x4b, 0x5f, 0x6e, 0x4d, 0x62, 0x6a,
  0x6f, 0x80, 0x8e, 0x6a, 0x7a, 0x89, 0x55, 0x62, 0x71, 0x46, 0x51, 0x5e,
  0x5d, 0x68, 0x74, 0x71, 0x81, 0x92, 0x5f, 0x71, 0x7f, 0x5d, 0x73, 0x7d,
  0x6c, 0x84, 0x94, 0x73, 0x8b, 0xa5, 0x6b, 0x86, 0xa5, 0x61, 0x7b, 0x97,
  0x62, 0x78, 0x8e, 0x5f, 0x73, 0x82, 0x62, 0x74, 0x7f, 0x61, 0x77, 0x7f,
  0x5a, 0x71, 0x7b, 0x55, 0x68, 0x72, 0x95, 0x9e, 0xa5, 0xbb, 0xbd, 0xbf,
  0xb3, 0xb7, 0xb4, 0x92, 0x94, 0x8e, 0x70, 0x6d, 0x64, 0xcc, 0xc5, 0xb8,
  0x9a, 0x95, 0x8c, 0x5f, 0x67, 0x6d, 0x57, 0x63, 0x6c, 0x55, 0x63, 0x6c,
  0x3a, 0x51, 0x64, 0x56, 0x6b, 0x82, 0x5e, 0x71, 0x84, 0x4a, 0x5d, 0x68,
  0x64, 0x75, 0x81, 0x64, 0x73, 0x82, 0x4d, 0x5d, 0x6d, 0x55, 0x66, 0x76,
  0x78, 0x89, 0x9a, 0x85, 0x98, 0xab, 0x7f, 0x95, 0xa8, 0x6c, 0x83, 0x97,
  0x69, 0x82, 0x9b, 0x6e, 0x88, 0xa5, 0x62, 0x7d, 0x9a, 0x57, 0x71, 0x8b,
  0x51, 0x6c, 0x82, 0x51, 0x6c, 0x7d, 0x57, 0x72, 0x81, 0x5f, 0x7b, 0x8c,
  0x5f, 0x76, 0x86, 0x70, 0x7f, 0x89, 0xaa, 0xaf, 0xb3, 0xc3, 0xc2, 0xc1,
  0xd0, 0xce, 0xcb, 0xc1, 0xbd, 0xb9, 0x7f, 0x76, 0x70, 0xad, 0xa2, 0x98,
  0xb2, 0xa7, 0xa0, 0x50, 0x50, 0x54, 0x55, 0x62, 0x69, 0x4f, 0x65, 0x6d,
  0x4a, 0x61, 0x78, 0x59, 0x6e, 0x86, 0x57, 0x6a, 0x7e, 0x4b, 0x5d, 0x6a,
  0x52, 0x62, 0x71, 0x44, 0x54, 0x65, 0x47, 0x59, 0x6c, 0x50, 0x64, 0x79,
  0x59, 0x6e, 0x83, 0x67, 0x7b, 0x91, 0x76, 0x8a, 0xa1, 0x6f, 0x86, 0x9f,
  0x65, 0x7e, 0x99, 0x6a, 0x83, 0xa1, 0x69, 0x82, 0x9e, 0x62, 0x7c, 0x95,
  0x60, 0x7b, 0x92, 0x62, 0x7f, 0x93, 0x62, 0x7e, 0x91, 0x6d, 0x87, 0x9b,
  0x72, 0x86, 0x96, 0x8e, 0x99, 0xa1, 0xb4, 0xb5, 0xb6, 0xb8, 0xb4, 0xb1,
  0xbf, 0xb8, 0xb5, 0xc0, 0xb7, 0xb5, 0xa0, 0x97, 0x94, 0x84, 0x7a, 0x76,
  0xaa, 0xa0, 0x9d, 0x50, 0x4d, 0x50, 0x3c, 0x48, 0x4e, 0x43, 0x59, 0x60,
  0x4d, 0x66, 0x7f, 0x4f, 0x65, 0x80, 0x52, 0x66, 0x7d, 0x4e, 0x61, 0x72,
  0x4f, 0x5f, 0x74, 0x48, 0x58, 0x71, 0x46, 0x58, 0x73, 0x56, 0x6b, 0x86,
  0x6d, 0x82, 0x9f, 0x79, 0x8e, 0xa8, 0x81, 0x97, 0xb0, 0x85, 0x9e, 0xb8,
  0x89, 0xa3, 0xbe, 0x88, 0xa4, 0xbf, 0x87, 0xa1, 0xbe, 0x83, 0x9d, 0xb9,
  0x92, 0xab, 0xc5, 0x94, 0xad, 0xc4, 0x92, 0xaa, 0xbe, 0x96, 0xa9, 0xb8,
  0x94, 0xa2, 0xac, 0xa3, 0xa9, 0xad, 0xb3, 0xb1, 0xaf, 0xb5, 0xae, 0xa8,
  0xb9, 0xb0, 0xab, 0xb0, 0xa8, 0xa5, 0xaa, 0xa4, 0xa3, 0x65, 0x62, 0x63,
  0x5a, 0x57, 0x57, 0x49, 0x46, 0x47, 0x37, 0x3f, 0x43, 0x3b, 0x4b, 0x51,
  0x60, 0x7b, 0x97, 0x5e, 0x76, 0x95, 0x6a, 0x7f, 0x9a, 0x68, 0x7c, 0x91,
  0x6d, 0x7e, 0x99, 0x83, 0x94, 0xb3, 0x84, 0x98, 0xb8, 0x8a, 0xa0, 0xc1,
  0x90, 0xa7, 0xc9, 0x98, 0xaf, 0xcd, 0x9b, 0xb3, 0xcf, 0x9a, 0xb4, 0xd0,
  0x9b, 0xb7, 0xd1, 0x9b, 0xb8, 0xd2, 0x9e, 0xb9, 0xd4, 0x94, 0xae, 0xc8,
  0x96, 0xad, 0xc6, 0x9d, 0xb2, 0xc9, 0x9c, 0xaf, 0xc3, 0x92, 0xa0, 0xab,
  0x77, 0x80, 0x86, 0x82, 0x85, 0x87, 0x92, 0x8e, 0x8b, 0xa9, 0xa1, 0x9b,
  0xb1, 0xa9, 0xa5, 0xa8, 0xa2, 0xa1, 0xa7, 0xa6, 0xa8, 0x69, 0x6c, 0x71,
  0x45, 0x46, 0x49, 0x62, 0x5f, 0x5c, 0x56, 0x59, 0x5a, 0x48, 0x52, 0x56,
  0x6a, 0x85, 0xa5, 0x65, 0x7d, 0x9f, 0x73, 0x8a, 0xa7, 0x83, 0x98, 0xb0,
  0x81, 0x94, 0xaf, 0x87, 0x9a, 0xb9, 0x90, 0xa6, 0xc6, 0x8f, 0xa7, 0xc9,
  0x92, 0xaa, 0xcd, 0x96, 0xae, 0xd0, 0x9a, 0xb3, 0xd2, 0x9a, 0xb6, 0xd2,
  0x9a, 0xb9, 0xd2, 0x99, 0xb9, 0xd0, 0x97, 0xb5, 0xc9, 0x90, 0xab, 0xbe,
  0x82, 0x98, 0xaa, 0x86, 0x98, 0xaa, 0x8b, 0x9a, 0xa9, 0x7f, 0x8b, 0x94,
  0x5e, 0x66, 0x6c, 0x75, 0x78, 0x7a, 0x92, 0x90, 0x8e, 0x9f, 0x9b, 0x99,
  0xa7, 0xa5, 0xa7, 0xa3, 0xa2, 0xa5, 0xa2, 0xa3, 0xa7, 0x84, 0x88, 0x8e,
  0x90, 0x91, 0x93, 0xc0, 0xbb, 0xb5, 0x9a, 0x99, 0x98, 0x69, 0x6c, 0x6f,
  0x5f, 0x7c, 0x9d, 0x6c, 0x85, 0xaa, 0x76, 0x8d, 0xae, 0x6d, 0x83, 0x9d,
  0x5f, 0x74, 0x8d, 0x5d, 0x72, 0x8f, 0x81, 0x98, 0xb6, 0x91, 0xaa, 0xca,
  0x95, 0xaf, 0xd0, 0x97, 0xaf, 0xd3, 0x96, 0xb0, 0xd1, 0x90, 0xad, 0xca,
  0x86, 0xa7, 0xbf, 0x7e, 0xa0, 0xb5, 0x7a, 0x9a, 0xa7, 0x7b, 0x97, 0xa2,
  0x7a, 0x90, 0x9b, 0x85, 0x96, 0xa0, 0x9b, 0xa8, 0xb2, 0x94, 0xa0, 0xa9,
  0x83, 0x8c, 0x93, 0x93, 0x98, 0x9d, 0xa2, 0xa5, 0xa7, 0x9c, 0x9e, 0xa0,
  0x9d, 0xa3, 0xa9, 0x97, 0x9b, 0xa2, 0x99, 0x9c, 0xa1, 0x95, 0x97, 0x9c,
  0x9f, 0x9e, 0x9e, 0xa4, 0x9e, 0x96, 0x9d, 0x98, 0x96, 0x94, 0x93, 0x94,
  0x66, 0x81, 0xa2, 0x59, 0x71, 0x94, 0x49, 0x60, 0x80, 0x46, 0x5d, 0x79,
  0x56, 0x6e, 0x8a, 0x6f, 0x87, 0xa4, 0x7b, 0x94, 0xb1, 0x8c, 0xa4, 0xc2,
  0x8f, 0xa9, 0xc7, 0x81, 0x9f, 0xbc, 0x78, 0x96, 0xb2, 0x75, 0x92, 0xab,
  0x78, 0x94, 0xaa, 0x7e, 0x99, 0xac, 0x85, 0x9f, 0xac, 0x8d, 0xa5, 0xb0,
  0x96, 0xaa, 0xb4, 0x8e, 0xa0, 0xa8, 0x99, 0xa8, 0xb0, 0x9f, 0xac, 0xb5,
  0x97, 0xa1, 0xaa, 0x9d, 0xa4, 0xad, 0xa5, 0xa8, 0xaf, 0xa1, 0xa4, 0xac,
  0x99, 0xa0, 0xa8, 0x98, 0x9e, 0xa5, 0x9a, 0x9e, 0xa3, 0x90, 0x92, 0x96,
  0x83, 0x84, 0x85, 0x79, 0x79, 0x76, 0x7d, 0x7d, 0x7d, 0x95, 0x96, 0x99,
  0x56, 0x70, 0x8f, 0x3d, 0x55, 0x75, 0x47, 0x60, 0x7f, 0x6e, 0x87, 0xa6,
  0x80, 0x9b, 0xbc, 0x8a, 0xa5, 0xc5, 0x82, 0x9c, 0xba, 0x7b, 0x93, 0xb0,
  0x76, 0x8e, 0xa9, 0x6c, 0x8d, 0xa3, 0x76, 0x97, 0xab, 0x84, 0xa0, 0xb2,
  0x8f, 0xa7, 0xb9, 0x98, 0xad, 0xbe, 0x9c, 0xae, 0xbf, 0x99, 0xab, 0xba,
  0x95, 0xa7, 0xb4, 0x89, 0x9b, 0xa5, 0x91, 0xa2, 0xac, 0x9a, 0xa8, 0xb1,
  0x99, 0xa3, 0xad, 0x9a, 0xa1, 0xac, 0xa0, 0xa3, 0xaf, 0xa4, 0xa6, 0xb2,
  0x98, 0x9e, 0xa4, 0x90, 0x97, 0x9c, 0x7d, 0x82, 0x88, 0x69, 0x6f, 0x74,
  0x5c, 0x62, 0x67, 0x4b, 0x52, 0x57, 0x56, 0x5c, 0x62, 0x84, 0x8a, 0x90,
  0x68, 0x85, 0xa6, 0x67, 0x84, 0xa5, 0x6b, 0x87, 0xa8, 0x72, 0x8f, 0xb0,
  0x73, 0x8f, 0xb0, 0x74, 0x8f, 0xae, 0x7b, 0x96, 0xb3, 0x76, 0x8e, 0xaa,
  0x74, 0x8a, 0xa4, 0x86, 0xa1, 0xb8, 0x8d, 0xa9, 0xbe, 0x90, 0xab, 0xbd,
  0x8f, 0xa9, 0xb9, 0x8d, 0xa5, 0xb5, 0x85, 0x97, 0xa8, 0x75, 0x86, 0x96,
  0x62, 0x74, 0x81, 0x59, 0x6b, 0x76, 0x82, 0x94, 0x9d, 0x96, 0xa4, 0xad,
  0x97, 0xa3, 0xad, 0x9a, 0xa3, 0xad, 0x98, 0x9f, 0xab, 0x91, 0x98, 0xa2,
  0x75, 0x80, 0x85, 0x60, 0x6c, 0x70, 0x5a, 0x66, 0x6a, 0x50, 0x5b, 0x5f,
  0x41, 0x4b, 0x50, 0x47, 0x4e, 0x54, 0x49, 0x50, 0x56, 0x41, 0x48, 0x4e,
  0x63, 0x84, 0xa5, 0x6b, 0x8c, 0xac, 0x6f, 0x90, 0xb1, 0x6f, 0x8f, 0xaf,
  0x72, 0x90, 0xae, 0x77, 0x93, 0xb0, 0x7d, 0x97, 0xb3, 0x7e, 0x96, 0xb0,
  0x75, 0x8b, 0xa4, 0x7d, 0x92, 0xab, 0x7d, 0x94, 0xab, 0x81, 0x9a, 0xad,
  0x83, 0x9f, 0xaf, 0x82, 0x9c, 0xab, 0x5b, 0x6e, 0x7e, 0x3d, 0x4e, 0x5e,
  0x39, 0x4b, 0x58, 0x38, 0x4a, 0x55, 0x73, 0x84, 0x8e, 0x94, 0xa3, 0xac,
  0x8b, 0x99, 0xa2, 0x82, 0x8f, 0x98, 0x72, 0x7e, 0x88, 0x5f, 0x6d, 0x75,
  0x56, 0x67, 0x6c, 0x53, 0x64, 0x68, 0x49, 0x5a, 0x5e, 0x3a, 0x4c, 0x50,
  0x3c, 0x4b, 0x4f, 0x4b, 0x52, 0x58, 0x33, 0x39, 0x3f, 0x1b, 0x21, 0x27,
  0x3e, 0x5e, 0x7c, 0x68, 0x89, 0xa6, 0x74, 0x94, 0xb2, 0x72, 0x92, 0xaf,
  0x74, 0x92, 0xaf, 0x75, 0x91, 0xad, 0x66, 0x81, 0x9a, 0x5b, 0x73, 0x8b,
  0x54, 0x6a, 0x82, 0x51, 0x66, 0x7e, 0x4e, 0x63, 0x7a, 0x70, 0x87, 0x9a,
  0x85, 0x9d, 0xae, 0x82, 0x9a, 0xa9, 0x60, 0x72, 0x83, 0x4c, 0x5d, 0x6c,
  0x53, 0x65, 0x72, 0x56, 0x68, 0x72, 0x6b, 0x7d, 0x86, 0x6c, 0x7c, 0x85,
  0x60, 0x71, 0x79, 0x58, 0x6a, 0x71, 0x53, 0x66, 0x6d, 0x51, 0x65, 0x6c,
  0x46, 0x58, 0x5f, 0x3d, 0x4f, 0x56, 0x33, 0x45, 0x4b, 0x2d, 0x3f, 0x45,
  0x34, 0x43, 0x49, 0x2e, 0x35, 0x3b, 0x1e, 0x25, 0x2b, 0x18, 0x1f, 0x26,
  0x39, 0x56, 0x6f, 0x60, 0x7d, 0x96, 0x6a, 0x86, 0xa0, 0x69, 0x86, 0x9f,
  0x6b, 0x89, 0xa3, 0x68, 0x85, 0x9e, 0x41, 0x5c, 0x74, 0x35, 0x4d, 0x64,
  0x3b, 0x52, 0x68, 0x40, 0x59, 0x70, 0x44, 0x5c, 0x73, 0x6e, 0x83, 0x97,
  0x87, 0x99, 0xac, 0x85, 0x96, 0xa8, 0x73, 0x84, 0x95, 0x62, 0x73, 0x83,
  0x58, 0x6a, 0x77, 0x4f, 0x61, 0x6b, 0x4e, 0x60, 0x6a, 0x50, 0x61, 0x6a,
  0x51, 0x63, 0x6a, 0x50, 0x65, 0x6c, 0x46, 0x5f, 0x64, 0x37, 0x4f, 0x54,
  0x2c, 0x3c, 0x45, 0x31, 0x40, 0x49, 0x2d, 0x3c, 0x44, 0x29, 0x39, 0x41,
  0x22, 0x2f, 0x37, 0x1e, 0x26, 0x2c, 0x1b, 0x22, 0x28, 0x18, 0x1f, 0x25,
  0x41, 0x5a, 0x71, 0x5a, 0x74, 0x8a, 0x68, 0x81, 0x97, 0x69, 0x83, 0x9a,
  0x6d, 0x89, 0xa1, 0x6d, 0x8a, 0xa2, 0x4f, 0x6a, 0x80, 0x49, 0x61, 0x77,
  0x55, 0x6c, 0x81, 0x58, 0x74, 0x88, 0x62, 0x7c, 0x90, 0x6a, 0x7e, 0x92,
  0x62, 0x71, 0x84, 0x53, 0x5f, 0x72, 0x44, 0x54, 0x65, 0x41, 0x53, 0x62,
  0x46, 0x59, 0x65, 0x4a, 0x5c, 0x67, 0x51, 0x62, 0x6d, 0x52, 0x64, 0x6c,
  0x48, 0x5c, 0x63, 0x33, 0x4a, 0x50, 0x29, 0x43, 0x48, 0x2c, 0x45, 0x4b,
  0x3d, 0x4a, 0x55, 0x37, 0x42, 0x4d, 0x27, 0x33, 0x3d, 0x23, 0x2e, 0x3a,
  0x20, 0x2b, 0x34, 0x1e, 0x26, 0x2d, 0x1b, 0x23, 0x29, 0x19, 0x20, 0x27,
  0x43, 0x5c, 0x74, 0x57, 0x70, 0x88, 0x69, 0x82, 0x9a, 0x67, 0x7f, 0x97,
  0x66, 0x7c, 0x94, 0x63, 0x78, 0x8f, 0x58, 0x6e, 0x84, 0x51, 0x66, 0x7c,
  0x4c, 0x61, 0x76, 0x45, 0x5a, 0x67, 0x3b, 0x50, 0x5b, 0x39, 0x4c, 0x57,
  0x3a, 0x4b, 0x56, 0x3f, 0x50, 0x5b, 0x42, 0x54, 0x63, 0x46, 0x5b, 0x65,
  0x48, 0x5f, 0x65, 0x44, 0x59, 0x63, 0x3e, 0x51, 0x5e, 0x36, 0x49, 0x52,
  0x2e, 0x41, 0x48, 0x2f, 0x41, 0x48, 0x31, 0x43, 0x4a, 0x2c, 0x3d, 0x45,
  0x38, 0x45, 0x4e, 0x2e, 0x3a, 0x44, 0x1e, 0x2a, 0x34, 0x1c, 0x28, 0x32,
  0x1d, 0x2a, 0x33, 0x19, 0x28, 0x30, 0x18, 0x23, 0x2f, 0x1e, 0x25, 0x30,
  0x36, 0x4c, 0x60, 0x3a, 0x4f, 0x64, 0x41, 0x56, 0x6a, 0x3a, 0x4e, 0x63,
  0x37, 0x4a, 0x5c, 0x32, 0x45, 0x56, 0x2d, 0x40, 0x52, 0x2c, 0x3f, 0x51,
  0x2e, 0x41, 0x52, 0x33, 0x46, 0x4f, 0x37, 0x4a, 0x51, 0x3a, 0x4d, 0x54,
  0x3e, 0x51, 0x57, 0x40, 0x52, 0x5b, 0x3e, 0x51, 0x5e, 0x3a, 0x50, 0x58,
  0x33, 0x49, 0x4d, 0x26, 0x3b, 0x44, 0x25, 0x37, 0x44, 0x30, 0x44, 0x4c,
  0x31, 0x43, 0x4a, 0x30, 0x3f, 0x48, 0x2a, 0x36, 0x40, 0x26, 0x32, 0x3b,
  0x29, 0x36, 0x3f, 0x20, 0x2d, 0x36, 0x1b, 0x28, 0x31, 0x1c, 0x28, 0x32,
  0x1b, 0x29, 0x31, 0x19, 0x28, 0x30, 0x1c, 0x27, 0x34, 0x1f, 0x25, 0x30,
  0x1e, 0x2d, 0x3b, 0x1d, 0x2b, 0x39, 0x1a, 0x28, 0x36, 0x1b, 0x2a, 0x37,
  0x1f, 0x31, 0x3c, 0x20, 0x34, 0x3d, 0x21, 0x35, 0x3e, 0x27, 0x3a, 0x44,
  0x31, 0x44, 0x4e, 0x34, 0x47, 0x4f, 0x35, 0x48, 0x4f, 0x33, 0x46, 0x4d,
  0x2e, 0x41, 0x48, 0x28, 0x3b, 0x44, 0x26, 0x3a, 0x45, 0x28, 0x3b, 0x42,
  0x26, 0x38, 0x3d, 0x2c, 0x3f, 0x47, 0x42, 0x55, 0x61, 0x37, 0x49, 0x52,
  0x29, 0x39, 0x41, 0x25, 0x33, 0x3c, 0x24, 0x2f, 0x38, 0x25, 0x2f, 0x39,
  0x1f, 0x2c, 0x35, 0x1b, 0x28, 0x31, 0x1a, 0x27, 0x30, 0x1b, 0x28, 0x31,
  0x1c, 0x29, 0x32, 0x1e, 0x2a, 0x32, 0x21, 0x2a, 0x32, 0x17, 0x1d, 0x23,
  0x21, 0x2b, 0x32, 0x1f, 0x28, 0x30, 0x1b, 0x24, 0x2c, 0x1c, 0x26, 0x2d,
  0x1c, 0x2c, 0x31, 0x1e, 0x30, 0x35, 0x1f, 0x31, 0x35, 0x20, 0x32, 0x37,
  0x23, 0x35, 0x3a, 0x21, 0x34, 0x3b, 0x1e, 0x31, 0x38, 0x1e, 0x31, 0x38,
  0x22, 0x35, 0x3c, 0x27, 0x3b, 0x42, 0x29, 0x3e, 0x47, 0x2d, 0x3d, 0x44,
  0x2a, 0x37, 0x3b, 0x34, 0x45, 0x4b, 0x49, 0x5e, 0x68, 0x31, 0x42, 0x4b,
  0x1e, 0x2c, 0x35, 0x23, 0x2e, 0x38, 0x26, 0x2f, 0x3a, 0x20, 0x28, 0x33,
  0x1b, 0x27, 0x31, 0x1a, 0x27, 0x30, 0x1b, 0x28, 0x31, 0x1d, 0x2a, 0x33,
  0x1e, 0x29, 0x32, 0x26, 0x2e, 0x35, 0x1a, 0x20, 0x25, 0x0d, 0x12, 0x14,
  0x1f, 0x28, 0x2d, 0x1e, 0x27, 0x2c, 0x1a, 0x23, 0x28, 0x1a, 0x24, 0x29,
  0x19, 0x27, 0x2c, 0x19, 0x29, 0x2e, 0x1a, 0x29, 0x2e, 0x1b, 0x2b, 0x30,
  0x1d, 0x2e, 0x33, 0x20, 0x33, 0x39, 0x25, 0x38, 0x3f, 0x28, 0x3a, 0x41,
  0x2a, 0x3d, 0x44, 0x29, 0x3d, 0x44, 0x28, 0x3c, 0x44, 0x2a, 0x37, 0x3d,
  0x27, 0x2e, 0x32, 0x2e, 0x3c, 0x42, 0x40, 0x55, 0x5d, 0x26, 0x36, 0x3f,
  0x1c, 0x28, 0x31, 0x24, 0x2e, 0x38, 0x1e, 0x25, 0x30, 0x1d, 0x24, 0x2f,
  0x1a, 0x26, 0x2f, 0x19, 0x26, 0x2f, 0x1b, 0x28, 0x31, 0x1c, 0x2a, 0x33,
  0x21, 0x2b, 0x34, 0x25, 0x28, 0x2e, 0x09, 0x0d, 0x0e, 0x04, 0x07, 0x05,
  0x17, 0x22, 0x27, 0x1b, 0x26, 0x2b, 0x19, 0x24, 0x29, 0x1c, 0x28, 0x2d,
  0x1e, 0x2c, 0x32, 0x20, 0x2f, 0x36, 0x22, 0x31, 0x38, 0x25, 0x33, 0x3b,
  0x27, 0x36, 0x3e, 0x27, 0x39, 0x40, 0x28, 0x3b, 0x42, 0x27, 0x39, 0x40,
  0x26, 0x39, 0x40, 0x23, 0x37, 0x3d, 0x1e, 0x33, 0x3a, 0x21, 0x2b, 0x31,
  0x1c, 0x1f, 0x24, 0x24, 0x30, 0x35, 0x39, 0x4e, 0x55, 0x24, 0x32, 0x3b,
  0x1e, 0x29, 0x32, 0x1d, 0x25, 0x30, 0x1d, 0x23, 0x2f, 0x1d, 0x23, 0x2f,
  0x18, 0x25, 0x2e, 0x18, 0x25, 0x2e, 0x17, 0x24, 0x2d, 0x1b, 0x29, 0x32,
  0x24, 0x2d, 0x36, 0x13, 0x14, 0x18, 0x04, 0x06, 0x03, 0x05, 0x07, 0x03,
  0x1c, 0x29, 0x2f, 0x1e, 0x2b, 0x32, 0x20, 0x2d, 0x34, 0x22, 0x2f, 0x36,
  0x21, 0x30, 0x38, 0x22, 0x30, 0x38, 0x23, 0x30, 0x38, 0x25, 0x31, 0x3a,
  0x26, 0x32, 0x3a, 0x26, 0x33, 0x37, 0x24, 0x31, 0x35, 0x22, 0x2d, 0x32,
  0x1e, 0x28, 0x2d, 0x18, 0x22, 0x27, 0x0f, 0x1a, 0x1e, 0x0c, 0x11, 0x14,
  0x08, 0x08, 0x0b, 0x13, 0x1b, 0x21, 0x2d, 0x3f, 0x48, 0x20, 0x2c, 0x36,
  0x19, 0x21, 0x2c, 0x1b, 0x22, 0x2d, 0x1b, 0x22, 0x2d, 0x1c, 0x23, 0x2e,
  0x18, 0x23, 0x2c, 0x15, 0x22, 0x2c, 0x14, 0x22, 0x2b, 0x22, 0x2c, 0x34,
  0x19, 0x1f, 0x25, 0x05, 0x06, 0x08, 0x04, 0x05, 0x03, 0x07, 0x08, 0x07
};
unsigned int ship_len = 3072;
